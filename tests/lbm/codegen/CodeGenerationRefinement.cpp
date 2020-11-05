//======================================================================================================================
//
//  This file is part of waLBerla. waLBerla is free software: you can
//  redistribute it and/or modify it under the terms of the GNU General Public
//  License as published by the Free Software Foundation, either version 3 of
//  the License, or (at your option) any later version.
//
//  waLBerla is distributed in the hope that it will be useful, but WITHOUT
//  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
//  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
//  for more details.
//
//  You should have received a copy of the GNU General Public License along
//  with waLBerla (see COPYING.txt). If not, see <http://www.gnu.org/licenses/>.
//
//! \file CodeGenerationRefinement.cpp
//! \ingroup lbm
//! \author Helen Schottenhamml <helen.schottenhamml@fau.de>
//
//======================================================================================================================

#include <blockforest/Initialization.h>
#include <blockforest/SetupBlockForest.h>
#include <blockforest/loadbalancing/StaticCurve.h>
#include <boundary/BoundaryHandling.h>
#include <core/Abort.h>
#include <core/SharedFunctor.h>
#include <core/debug/Debug.h>
#include <core/debug/TestSubsystem.h>
#include <core/math/Limits.h>
#include <core/mpi/Environment.h>
#include <core/mpi/MPIManager.h>
#include <core/timing/RemainingTimeLogger.h>
#include <domain_decomposition/SharedSweep.h>
#include <field/AddToStorage.h>
#include <field/FlagField.h>
#include <lbm/boundary/NoSlip.h>
#include <lbm/boundary/SimplePressure.h>
#include <lbm/boundary/UBB.h>

#include <lbm/field/AddToStorage.h>
#include <lbm/field/PdfField.h>
#include <lbm/lattice_model/D3Q19.h>
#include <lbm/lattice_model/D3Q27.h>
#include <lbm/sweeps/CellwiseSweep.h>
#include <timeloop/SweepTimeloop.h>
#include <type_traits>

#include <lbm/refinement_rebase/TimeStep.h>

// #include "CodeGenerationRefinement_D3Q19_SRT_COMP_LatticeModel.h"
#include "CodeGenerationRefinement_D3Q19_SRT_INCOMP_LatticeModel.h"
// #include "CodeGenerationRefinement_D3Q27_SRT_COMP_LatticeModel.h"
// #include "CodeGenerationRefinement_D3Q27_SRT_INCOMP_LatticeModel.h"
// #include "CodeGenerationRefinement_D3Q27_TRT_INCOMP_LatticeModel.h"
// #include "CodeGenerationRefinement_D3Q19_MRT_COMP_LatticeModel.h"

#define TEST_USES_VTK_OUTPUT
#ifdef TEST_USES_VTK_OUTPUT
#include "lbm/vtk/Density.h"
#include "lbm/vtk/Velocity.h"
#include "vtk/VTKOutput.h"
#endif

//#define DEVEL_OUTPUT
#ifdef DEVEL_OUTPUT
#include "core/math/Sample.h"
#endif

namespace walberla {

using flag_t = walberla::uint64_t;
using FlagField_T = FlagField<flag_t>;

const FlagUID          Fluid_Flag( "fluid" );
const FlagUID            UBB_Flag( "velocity bounce back" );
const FlagUID         NoSlip_Flag( "no slip" );
const FlagUID SimplePressure_Flag( "simple pressure" );

const uint_t FieldGhostLayers = uint_t(4);
const real_t GlobalOmega      = real_t(1.4);


/////////////////////
// BLOCK STRUCTURE //
/////////////////////

static void refinementSelection( SetupBlockForest& forest, const uint_t levels )
{
   const AABB & domain = forest.getDomain();

   const real_t xMid = (domain.xMax() - domain.xMin()) / real_t(2.0);
   const real_t yMid = (domain.yMax() - domain.yMin()) / real_t(2.0);
   const real_t zMid = (domain.zMax() - domain.zMin()) / real_t(2.0);

   const real_t span  = domain.ySize() / real_t(64);

   AABB refinementBox( xMid - span, yMid - span, zMid - span, xMid + span, yMid + span, zMid + span );
//   AABB refinementBox( domain.xMin(), domain.yMax() - real_t(2.0) * span, domain.zMin(), domain.xMax(), domain.yMax(), domain.zMax() );

   for( auto block = forest.begin(); block != forest.end(); ++block )
   {
      if( block->getAABB().intersects(refinementBox) )
         if( block->getLevel() < ( levels - uint_t(1) ) )
            block->setMarker( true );
   }
}

static void workloadAndMemoryAssignment( SetupBlockForest& forest )
{
   for( auto block = forest.begin(); block != forest.end(); ++block )
   {
      block->setWorkload( numeric_cast< workload_t >( uint_t(1) << block->getLevel() ) );
      block->setMemory( numeric_cast< memory_t >(1) );
   }
}

static shared_ptr< StructuredBlockForest > createBlockStructure( const uint_t levels,
                                                                 const uint_t numberOfXBlocks,        const uint_t numberOfYBlocks,        const uint_t numberOfZBlocks,
                                                                 const uint_t numberOfXCellsPerBlock, const uint_t numberOfYCellsPerBlock, const uint_t numberOfZCellsPerBlock,
                                                                 const bool   xPeriodic = false,      const bool   yPeriodic = false,      const bool   zPeriodic = false,
                                                                 const bool keepGlobalBlockInformation = false )
{
   // initialize SetupBlockForest = determine domain decomposition
   SetupBlockForest sforest;

   sforest.addRefinementSelectionFunction( std::bind( refinementSelection, std::placeholders::_1, levels ) );
   sforest.addWorkloadMemorySUIDAssignmentFunction( workloadAndMemoryAssignment );

   sforest.init( AABB( real_c(0), real_c(0), real_c(0), real_c( numberOfXBlocks * numberOfXCellsPerBlock ),
                       real_c( numberOfYBlocks * numberOfYCellsPerBlock ),
                       real_c( numberOfZBlocks * numberOfZCellsPerBlock ) ),
                 numberOfXBlocks, numberOfYBlocks, numberOfZBlocks, xPeriodic, yPeriodic, zPeriodic );

   // calculate process distribution
   const memory_t memoryLimit = math::Limits< memory_t >::inf();

   sforest.balanceLoad( blockforest::StaticLevelwiseCurveBalance(true), uint_c( MPIManager::instance()->numProcesses() ), real_t(0), memoryLimit, true );

   sforest.writeVTKOutput("domain_decomposition");

   WALBERLA_LOG_INFO_ON_ROOT( sforest );

   MPIManager::instance()->useWorldComm();

   // create StructuredBlockForest (encapsulates a newly created BlockForest)
   shared_ptr< StructuredBlockForest > sbf =
      make_shared< StructuredBlockForest >( make_shared< BlockForest >( uint_c( MPIManager::instance()->rank() ), sforest, keepGlobalBlockInformation ),
                                            numberOfXCellsPerBlock, numberOfYCellsPerBlock, numberOfZCellsPerBlock );
   sbf->createCellBoundingBoxes();
   return sbf;
}

template< typename LatticeModel_T, typename BoundaryHandling_T >
void initialiseDomain(std::shared_ptr<StructuredBlockForest> & storage, const BlockDataID & pdfFieldID, const BlockDataID & boundaryHandlingID, const real_t & maxVel) {

   const Vector3<real_t> maxPoint = storage->getDomain().max();

   for( auto blockIt = storage->begin(); blockIt != storage->end(); ++blockIt )
   {
      auto handling = blockIt->template getData<BoundaryHandling_T> ( boundaryHandlingID );
      auto pdfField = blockIt->template getData< lbm::PdfField<LatticeModel_T> > ( pdfFieldID );
      auto flagField = handling->getFlagField();

      uint_t level = storage->getLevel(*blockIt);
      real_t factor = 1.0 / real_t((1 << level));

      Cell offset(0,0,0);

      storage->transformBlockLocalToGlobalCell( offset, *blockIt );

      Cell levelOffset(uint_t(factor * offset.x()), uint_t(factor * offset.y()), uint_t(factor * offset.z()));

      // go over inner part - if velocity 0 -> set to noslip boundary
      for( auto cellIt = flagField->beginXYZ(); cellIt != flagField->end(); ++cellIt )
      {
         auto cell = cellIt.cell();
         if ( !handling->isBoundary( cell ) )
         {
            real_t y = cell[1] * storage->dx(level) / storage->dy(0) + levelOffset[1] + 0.5 * storage->dy(level);

            const real_t max = maxPoint[1];
            const real_t velocity = - real_t(4) * y * (y-max) / ( max * max) * maxVel;

            real_t density = pdfField->getDensity( cell ); // Preserve old density value
            pdfField->setDensityAndVelocity( cell, Vector3<real_t>(velocity, 0, 0), density );
         }
      }
   }

}

template< typename LatticeModel_T >
class WalberlaBoundaryHandling
{
 public:

   typedef lbm::NoSlip< LatticeModel_T, flag_t >         NoSlip_T;
   typedef lbm::UBB< LatticeModel_T, flag_t >            UBB_T;
   typedef lbm::SimplePressure< LatticeModel_T, flag_t > SimplePressure_T;

   typedef BoundaryHandling< FlagField_T, typename LatticeModel_T::Stencil, NoSlip_T, UBB_T, SimplePressure_T > BoundaryHandling_T;

   WalberlaBoundaryHandling( const std::string & id, const BlockDataID & flagField, const BlockDataID & pdfField, const real_t velocity ) :
      id_( id ), flagField_( flagField ), pdfField_( pdfField ), velocity_( velocity )
   {}

   BoundaryHandling_T * operator()( IBlock* const block, const StructuredBlockStorage* const storage ) const;

 private:

   const std::string id_;

   const BlockDataID flagField_;
   const BlockDataID  pdfField_;

   const real_t velocity_;

};

template< typename LatticeModel_T >
typename WalberlaBoundaryHandling<LatticeModel_T>::BoundaryHandling_T *
WalberlaBoundaryHandling<LatticeModel_T>::operator()( IBlock * const block, const StructuredBlockStorage * const storage ) const
{
   using PdfField_T = lbm::PdfField< LatticeModel_T >;

   WALBERLA_ASSERT_NOT_NULLPTR( block );
   WALBERLA_ASSERT_NOT_NULLPTR( storage );

   FlagField_T * flagField = block->getData< FlagField_T >( flagField_ );
   PdfField_T *   pdfField = block->getData< PdfField_T > (  pdfField_ );

   const auto fluid = flagField->getOrRegisterFlag(Fluid_Flag);

   BoundaryHandling_T * handling = new BoundaryHandling_T( std::string("boundary handling ")+id_, flagField, fluid,
                                                           NoSlip_T( "no slip", NoSlip_Flag, pdfField ),
                                                           UBB_T( "velocity bounce back", UBB_Flag, pdfField ),
                                                           SimplePressure_T( "simple pressure", SimplePressure_Flag, pdfField, real_c(1.0) ));

   const uint_t level = storage->getLevel(*block);
   CellInterval domainBB = storage->getDomainCellBB(level);
   storage->transformGlobalToBlockLocalCellInterval(domainBB, *block);

   const real_t max = storage->getDomain().xSize();

   auto ghost = cell_idx_t(FieldGhostLayers);
   domainBB.expand( ghost );

   // west INFLOW
   CellInterval west( domainBB.xMin(), domainBB.yMin(), domainBB.zMin(),
                      domainBB.xMin() + ghost - cell_idx_c(1), domainBB.yMax(), domainBB.zMax());

   Cell offset(0,0,0);
   storage->transformBlockLocalToGlobalCell(offset, *block);

   //NOTE this for loop assumes west.empty()=false to avoid strict-overflow warning for gcc_7_hybrid
   for( auto cellIt = cell::CellIntervalIterator(west, west.min()); cellIt != west.end(); ++cellIt ) {
      const cell_idx_t x = cellIt->x();
      const cell_idx_t y = cellIt->y();
      const cell_idx_t z = cellIt->z();

      Cell globalCell = *cellIt + offset;

      const real_t coordY = globalCell[1] + 0.5 * storage->dy(level);

      const real_t velocity = - real_t(4) * coordY * (coordY-max) / ( max * max) * velocity_;

      handling->forceBoundary(UBB_Flag, x, y, z, typename UBB_T::Velocity(velocity, real_t(0.0), real_t(0.0)));
   }

   // east OUTFLOW
   CellInterval east( domainBB.xMax() - ghost + cell_idx_c(1), domainBB.yMin(), domainBB.zMin(),
                      domainBB.xMax(), domainBB.yMax(), domainBB.zMax() );
   handling->forceBoundary( SimplePressure_Flag, east );

   // no slip BOTTOM
   CellInterval bottom( domainBB.xMin(), domainBB.yMin(), domainBB.zMin(),
                        domainBB.xMax(), domainBB.yMin() + ghost - cell_idx_c(1), domainBB.zMax());
   handling->forceBoundary( NoSlip_Flag, bottom );

   // no slip TOP
   CellInterval top( domainBB.xMin(), domainBB.yMax() - ghost + cell_idx_c(1), domainBB.zMin(),
                     domainBB.xMax(), domainBB.yMax(), domainBB.zMax() );
   handling->forceBoundary( NoSlip_Flag, top );

   // FRONT and BACK periodic

   handling->fillWithDomain( domainBB );

   return handling;
}


template<typename... Ts>
using void_t = void;

template< typename T, typename = void >
struct is_waLBerlaLatticeModel : std::false_type{};

template< typename T >
struct is_waLBerlaLatticeModel<T, void_t<typename T::CollisionModel>> : std::true_type {};


template< typename LatticeModel_T, class Enable = void >
struct AddRefinementTimeStep;

// waLBerla Lattice Model
template< typename LatticeModel_T >
struct AddRefinementTimeStep < LatticeModel_T, typename std::enable_if< is_waLBerlaLatticeModel<LatticeModel_T>::value >::type >
{
   static void add( shared_ptr< StructuredBlockForest > & blocks, SweepTimeloop & timeloop, const LatticeModel_T & latticeModel, std::vector< BlockDataID > & fieldIds,
                    field::Layout layout, const BlockDataID & flagFieldId, const real_t velocity, const char * fieldName ) {

      fieldIds.push_back( lbm::addPdfFieldToStorage( blocks, std::string("pdf field ") + std::string(fieldName),
                                                     latticeModel, Vector3<real_t>( velocity, velocity / real_t(2), velocity / real_t(4) ), real_t(1),
                                                     FieldGhostLayers, layout ) );

      typedef typename WalberlaBoundaryHandling<LatticeModel_T>::BoundaryHandling_T BoundaryHandling_T;

      BlockDataID boundaryHandlingId = blocks->addStructuredBlockData< BoundaryHandling_T >(
         WalberlaBoundaryHandling< LatticeModel_T >(fieldName, flagFieldId, fieldIds.back(), velocity),
         std::string("boundary handling ") + std::string(fieldName));

      initialiseDomain<LatticeModel_T, BoundaryHandling_T>(blocks, fieldIds.back(), boundaryHandlingId, velocity);

      auto sweep = lbm::makeCellwiseSweep< LatticeModel_T, FlagField_T >( fieldIds.back(), flagFieldId, Fluid_Flag );

      auto refinementTimeStep = lbm::refinement::makeTimeStep<LatticeModel_T, BoundaryHandling_T>(
         blocks, sweep, fieldIds.back(), boundaryHandlingId
      );

      timeloop.addFuncBeforeTimeStep(makeSharedFunctor( refinementTimeStep ), "Refinement Time Step - " + std::string(fieldName));

#ifdef TEST_USES_VTK_OUTPUT

      auto vtkOutput  = vtk::createVTKOutput_BlockData(blocks, "fluid_field " + std::string(fieldName), uint_c(1), uint_c(0),
                                                       false, "vtk_out", "simulation_step", false, true, true, false, 0);

      vtkOutput->addCellDataWriter( std::make_shared<lbm::DensityVTKWriter<LatticeModel_T, float>>(fieldIds.back(), "Density (Lattice)"));
      vtkOutput->addCellDataWriter( std::make_shared<lbm::VelocityVTKWriter<LatticeModel_T, float>>(fieldIds.back(), "Velocity (Lattice)"));

      timeloop.addFuncAfterTimeStep( vtk::writeFiles( vtkOutput ), "VTK" );
#endif
   }
};

// lbmpy Lattice Model
template< typename LatticeModel_T >
struct AddRefinementTimeStep < LatticeModel_T, typename std::enable_if< !is_waLBerlaLatticeModel<LatticeModel_T>::value >::type >
{
   static void add( shared_ptr< StructuredBlockForest > & blocks, SweepTimeloop & timeloop, const LatticeModel_T & latticeModel, std::vector< BlockDataID > & fieldIds,
                    field::Layout layout, const BlockDataID & flagFieldId, const real_t velocity, const char * fieldName ) {

      fieldIds.push_back( lbm::addPdfFieldToStorage( blocks, std::string("pdf field ") + std::string(fieldName),
                                                     latticeModel, Vector3<real_t>( velocity, velocity / real_t(2), velocity / real_t(4) ), real_t(1),
                                                     FieldGhostLayers, layout ) );

      // TODO  currently we are forced to use the waLBerla functionalities for boundary handling
      //       -> generated or compiled-in are not possible
      typedef typename WalberlaBoundaryHandling<LatticeModel_T>::BoundaryHandling_T BoundaryHandling_T;

      BlockDataID boundaryHandlingId = blocks->addStructuredBlockData< BoundaryHandling_T >(
         WalberlaBoundaryHandling< LatticeModel_T >(fieldName, flagFieldId, fieldIds.back(), velocity),
         std::string("boundary handling ") + std::string(fieldName));

      initialiseDomain<LatticeModel_T, BoundaryHandling_T>(blocks, fieldIds.back(), boundaryHandlingId, velocity);

      auto sweep = std::make_shared<typename LatticeModel_T::Sweep>(typename LatticeModel_T::Sweep(fieldIds.back()));

      auto refinementTimeStep = lbm::refinement::makeTimeStep<LatticeModel_T, BoundaryHandling_T>(
         blocks, sweep, fieldIds.back(), boundaryHandlingId
      );

      timeloop.addFuncBeforeTimeStep(makeSharedFunctor( refinementTimeStep ), "Refinement Time Step - " + std::string(fieldName));

#ifdef TEST_USES_VTK_OUTPUT

      auto vtkOutput  = vtk::createVTKOutput_BlockData(blocks, "fluid_field " + std::string(fieldName), uint_c(1), uint_c(0),
                                                       false, "vtk_out", "simulation_step", false, true, true, false, 0);

      vtkOutput->addCellDataWriter(std::make_shared<lbm::DensityVTKWriter<LatticeModel_T, float>>(fieldIds.back(), "Density (Lattice)"));
      vtkOutput->addCellDataWriter( std::make_shared<lbm::VelocityVTKWriter<LatticeModel_T, float>>(fieldIds.back(), "Velocity (Lattice)"));

      timeloop.addFuncAfterTimeStep( vtk::writeFiles( vtkOutput ), "VTK" );
#endif
   }
};


//////////////////
/// EVALUATION ///
//////////////////

template< typename LatticeModel1_T, typename LatticeModel2_T >
void check( const shared_ptr< StructuredBlockForest > & blocks, const BlockDataID & fieldId1, const BlockDataID & fieldId2 )
{
   using PdfField1_T = lbm::PdfField< LatticeModel1_T >;
   using PdfField2_T = lbm::PdfField< LatticeModel2_T >;

   for( auto block = blocks->begin(); block != blocks->end(); ++block )
   {
      PdfField1_T * referenceField = block->template getData< PdfField1_T >( fieldId1 );
      PdfField2_T *          field = block->template getData< PdfField2_T >( fieldId2 );

      const auto & id1 = blocks->getBlockDataIdentifier( fieldId1 );
      const auto & id2 = blocks->getBlockDataIdentifier( fieldId2 );

      const std::string msg = "Check failed for fields with block data ID \"" + id1 + "\" and \"" + id2 + "\"";

#ifdef DEVEL_OUTPUT
      math::Sample samples[4];
#endif

      const auto xyz = referenceField->xyzSize();

      for( auto cell : xyz ) {

         Vector3< real_t > velocityReference;
         Vector3< real_t > velocity;

         real_t rhoReference = referenceField->getDensityAndVelocity( velocityReference, cell );
         real_t rho          =          field->getDensityAndVelocity( velocity         , cell );

#ifdef DEVEL_OUTPUT
         samples[0].insert( std::fabs( velocityReference[0] - velocity[0] ) );
         samples[1].insert( std::fabs( velocityReference[1] - velocity[1] ) );
         samples[2].insert( std::fabs( velocityReference[2] - velocity[2] ) );
         samples[3].insert( std::fabs( rhoReference - rho ) );
#else
         const real_t tol = 1e-5;

         WALBERLA_CHECK_FLOAT_EQUAL_EPSILON( velocityReference[0], velocity[0], tol, msg );
         WALBERLA_CHECK_FLOAT_EQUAL_EPSILON( velocityReference[1], velocity[1], tol, msg );
         WALBERLA_CHECK_FLOAT_EQUAL_EPSILON( velocityReference[2], velocity[2], tol, msg );
         WALBERLA_CHECK_FLOAT_EQUAL_EPSILON( rhoReference, rho, tol, msg );
#endif

      }

#ifdef DEVEL_OUTPUT
      WALBERLA_LOG_DEVEL( "Velocity (x): " << samples[0].format( "[%min, %max], %mean, %med" ) );
      WALBERLA_CHECK_FLOAT_EQUAL_EPSILON( samples[0].range(), 0.0, tol, msg );
      WALBERLA_LOG_DEVEL( "Velocity (y): " << samples[1].format( "[%min, %max], %mean, %med" ) );
      WALBERLA_CHECK_FLOAT_EQUAL_EPSILON( samples[1].range(), 0.0, tol, msg );
      WALBERLA_LOG_DEVEL( "Velocity (z): " << samples[2].format( "[%min, %max], %mean, %med" ) );
      WALBERLA_CHECK_FLOAT_EQUAL_EPSILON( samples[2].range(), 0.0, tol, msg );
      WALBERLA_LOG_DEVEL( "Density: " << samples[3].format( "[%min, %max], %mean, %med" ) );
      WALBERLA_CHECK_FLOAT_EQUAL_EPSILON( samples[3].range(), 0.0, tol, msg );
#endif
   }
}



int main( int argc, char ** argv )
{
   debug::enterTestMode();

   mpi::Environment env( argc, argv );

   const uint_t levels = uint_t(3);

   const uint_t xBlocks = uint_t(4);
   const uint_t yBlocks = uint_t(4);
   const uint_t zBlocks = uint_t(4);

   const uint_t xCells = uint_t(10);
   const uint_t yCells = uint_t(10);
   const uint_t zCells = uint_t(10);

   auto blocks = createBlockStructure( levels, xBlocks, yBlocks, zBlocks, xCells, yCells, zCells, false, false, true );

   BlockDataID flagFieldId = field::addFlagFieldToStorage< FlagField_T >( blocks, "flag field", FieldGhostLayers );

   SweepTimeloop timeloop( blocks->getBlockStorage(), uint_t(10) );

   std::vector<std::vector< BlockDataID >> fieldIds;

   const real_t velocity = real_t(0.05);

   ///////////////////////
   /// Collision Model ///
   ///////////////////////

   using SRT = lbm::collision_model::SRT;

   ////////////////////////////////
   /// D3Q19 SRT incompressible ///
   ////////////////////////////////

   fieldIds.emplace_back();

   typedef lbm::D3Q19<SRT, false>                                       walberalLM_D3Q19_SRT_INCOMP;
   typedef lbm::CodeGenerationRefinement_D3Q19_SRT_INCOMP_LatticeModel     lbmpyLM_D3Q19_SRT_INCOMP;

   AddRefinementTimeStep< walberalLM_D3Q19_SRT_INCOMP >::add(blocks, timeloop, walberalLM_D3Q19_SRT_INCOMP(SRT(GlobalOmega)),
                                                             fieldIds.back(), field::fzyx, flagFieldId, velocity, "( waLBerla D3Q19 SRT INCOMP )");
   AddRefinementTimeStep<    lbmpyLM_D3Q19_SRT_INCOMP >::add(blocks, timeloop,  lbmpyLM_D3Q19_SRT_INCOMP(GlobalOmega),
                                                             fieldIds.back(), field::fzyx, flagFieldId, velocity, "(   lbmpy  D3Q19 SRT INCOMP )");

   ////////////////////////////////
   /// D3Q19 SRT   compressible ///
   ////////////////////////////////

   fieldIds.emplace_back();

//   typedef lbm::D3Q19<SRT, true>                                      walberalLM_D3Q19_SRT_COMP;
//   typedef lbm::CodeGenerationRefinement_D3Q19_SRT_COMP_LatticeModel     lbmpyLM_D3Q19_SRT_COMP;
//
//   AddRefinementTimeStep< walberalLM_D3Q19_SRT_COMP >::add(blocks, timeloop, walberalLM_D3Q19_SRT_COMP(SRT(GlobalOmega)),
//                                                           fieldIds.back(), field::fzyx, flagFieldId, velocity, "( waLBerla D3Q19 SRT   COMP )");
//   AddRefinementTimeStep<    lbmpyLM_D3Q19_SRT_COMP >::add(blocks, timeloop,  lbmpyLM_D3Q19_SRT_COMP(GlobalOmega),
//                                                           fieldIds.back(), field::fzyx, flagFieldId, velocity, "(   lbmpy  D3Q19 SRT   COMP )");

   ////////////////////////////////
   /// D3Q27 SRT incompressible ///
   ////////////////////////////////

   fieldIds.emplace_back();

//   typedef lbm::D3Q27<SRT, false>                                       walberalLM_D3Q27_SRT_INCOMP;
//   typedef lbm::CodeGenerationRefinement_D3Q27_SRT_INCOMP_LatticeModel     lbmpyLM_D3Q27_SRT_INCOMP;
//
//   AddRefinementTimeStep< walberalLM_D3Q27_SRT_INCOMP >::add(blocks, timeloop, walberalLM_D3Q27_SRT_INCOMP(SRT(GlobalOmega)),
//                                                             fieldIds.back(), field::fzyx, flagFieldId, velocity, "( waLBerla D3Q27 SRT INCOMP )");
//   AddRefinementTimeStep<    lbmpyLM_D3Q27_SRT_INCOMP >::add(blocks, timeloop,  lbmpyLM_D3Q27_SRT_INCOMP(GlobalOmega),
//                                                             fieldIds.back(), field::fzyx, flagFieldId, velocity, "(   lbmpy  D3Q27 SRT INCOMP )");

   ////////////////////////////////
   /// D3Q27 SRT   compressible ///
   ////////////////////////////////

   fieldIds.emplace_back();

//   typedef lbm::D3Q27<SRT, true>                                      walberalLM_D3Q27_SRT_COMP;
//   typedef lbm::CodeGenerationRefinement_D3Q27_SRT_COMP_LatticeModel     lbmpyLM_D3Q27_SRT_COMP;
//
//   AddRefinementTimeStep< walberalLM_D3Q27_SRT_COMP >::add(blocks, timeloop, walberalLM_D3Q27_SRT_COMP(SRT(GlobalOmega)),
//                                                             fieldIds.back(), field::fzyx, flagFieldId, velocity, "( waLBerla D3Q27 SRT   COMP )");
//   AddRefinementTimeStep<    lbmpyLM_D3Q27_SRT_COMP >::add(blocks, timeloop,  lbmpyLM_D3Q27_SRT_COMP(GlobalOmega),
//                                                             fieldIds.back(), field::fzyx, flagFieldId, velocity, "(   lbmpy  D3Q27 SRT   COMP )");

   /////////////////////
   /// RUN TEST RUNS ///
   /////////////////////

   timeloop.addFuncAfterTimeStep(timing::RemainingTimeLogger( timeloop.getNrOfTimeSteps(), real_c(5.0) ),"remaining time logger");

   WcTimingPool timeloopTiming;
   timeloop.run( timeloopTiming );
   timeloopTiming.logResultOnRoot();

   //////////////////
   /// D3Q19, SRT ///
   //////////////////

   check<walberalLM_D3Q19_SRT_INCOMP, lbmpyLM_D3Q19_SRT_INCOMP>( blocks, fieldIds[0][0], fieldIds[0][1] );
   // check<walberalLM_D3Q19_SRT_COMP,   lbmpyLM_D3Q19_SRT_COMP>  ( blocks, fieldIds[1][0], fieldIds[1][1] );

   //////////////////
   /// D3Q27, SRT ///
   //////////////////

   // check<walberalLM_D3Q27_SRT_INCOMP, lbmpyLM_D3Q27_SRT_INCOMP>( blocks, fieldIds[2][0], fieldIds[2][1] );
   // check<walberalLM_D3Q27_SRT_COMP,   lbmpyLM_D3Q27_SRT_COMP>  ( blocks, fieldIds[3][0], fieldIds[3][1] );

   return EXIT_SUCCESS;
}
} // namespace walberla

int main( int argc, char* argv[] )
{
   return walberla::main( argc, argv );
}
