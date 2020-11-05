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
//! \file NonUniformBufferedSchemeTest.cpp
//! \ingroup comm
//! \author Helen Schottenhamml <helen.schottenhamml@fau.de>
//! \brief Checks communication for non-uniform buffered communication with field::refinement::PackInfo
//
//======================================================================================================================


#include <blockforest/communication/NonUniformBufferedScheme.h>
#include <blockforest/SetupBlockForest.h>
#include <blockforest/StructuredBlockForest.h>
#include <blockforest/loadbalancing/StaticCurve.h>

#include <boundary/BoundaryHandling.h>

#include <core/DataTypes.h>
#include <core/debug/TestSubsystem.h>
#include <core/logging/Logging.h>
#include <core/math/Limits.h>
#include <core/math/Sample.h>
#include <core/mpi/Environment.h>

#include <field/AddToStorage.h>
#include <field/iterators/FieldIterator.h>
#include <field/refinement/all.h>

#include <stencil/all.h>

#include <cstdlib>
#include <functional>



namespace nonuniform_buffered_scheme_test {

///////////
// USING //
///////////

using namespace walberla;
using walberla::real_t;
using walberla::uint_t;

//////////////
// TYPEDEFS //
//////////////

using CommunicationStencil_T = stencil::D3Q27;

const uint_t FieldGhostLayers = 4;

using ScalarField_T = field::GhostLayerField<real_t,1>;
using VectorField_T = field::GhostLayerField<Vector3<real_t>,1>;
using MultiComponentField_T = field::GhostLayerField<real_t,3>;

/////////////////////
// BLOCK STRUCTURE //
/////////////////////

static void refinementSelection( SetupBlockForest& forest, const uint_t levels )
{
   const AABB & domain = forest.getDomain();

   const real_t xSpan = domain.xSize() / real_t(32);
   const real_t ySpan = domain.ySize() / real_t(32);
   const real_t zSpan = domain.zSize() / real_t(64);

   const real_t xMiddle = ( domain.xMin() + domain.xMax() ) / real_t(2);
   const real_t yMiddle = ( domain.yMin() + domain.yMax() ) / real_t(2);
   const real_t zMiddle = ( domain.zMin() + domain.zMax() ) / real_t(2);

   AABB middleBox( xMiddle - xSpan, yMiddle - ySpan, zMiddle +             zSpan,
                   xMiddle + xSpan, yMiddle + ySpan, zMiddle + real_t(3) * zSpan );

   AABB shiftedBox( xMiddle +             xSpan, yMiddle +             ySpan, zMiddle +             zSpan,
                    xMiddle + real_t(3) * xSpan, yMiddle + real_t(3) * ySpan, zMiddle + real_t(3) * zSpan );

   for( auto block = forest.begin(); block != forest.end(); ++block )
   {
      if( block->getAABB().intersects( middleBox ) || block->getAABB().intersects( shiftedBox ) )
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
                                                                 const bool keepGlobalBlockInformation = false )
{
   // initialize SetupBlockForest = determine domain decomposition
   SetupBlockForest sforest;

   sforest.addRefinementSelectionFunction( std::bind( refinementSelection, std::placeholders::_1, levels ) );
   sforest.addWorkloadMemorySUIDAssignmentFunction( workloadAndMemoryAssignment );

   sforest.init( AABB( real_c(0), real_c(0), real_c(0), real_c( numberOfXBlocks * numberOfXCellsPerBlock ),
                       real_c( numberOfYBlocks * numberOfYCellsPerBlock ),
                       real_c( numberOfZBlocks * numberOfZCellsPerBlock ) ),
                 numberOfXBlocks, numberOfYBlocks, numberOfZBlocks, false, false, false );

   // calculate process distribution
   const memory_t memoryLimit = math::Limits< memory_t >::inf();

   sforest.balanceLoad( blockforest::StaticLevelwiseCurveBalance(true), uint_c( MPIManager::instance()->numProcesses() ), real_t(0), memoryLimit, true );

   MPIManager::instance()->useWorldComm();

   // create StructuredBlockForest (encapsulates a newly created BlockForest)
   shared_ptr< StructuredBlockForest > sbf =
      make_shared< StructuredBlockForest >( make_shared< BlockForest >( uint_c( MPIManager::instance()->rank() ), sforest, keepGlobalBlockInformation ),
                                            numberOfXCellsPerBlock, numberOfYCellsPerBlock, numberOfZCellsPerBlock );
   sbf->createCellBoundingBoxes();
   return sbf;
}

///////////////////////
// FIELD INITIALISER //
///////////////////////

template< typename Field_T >
void clearField( std::shared_ptr<StructuredBlockForest> & sbf, const BlockDataID & fieldID ) {

   for( auto it = sbf->begin(); it != sbf->end(); ++it ) {

      auto field = it->getData<Field_T>( fieldID );

      WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(
         field,
         for( uint_t f = 0; f < field->fSize(); ++f ) {
            field->get(x,y,z,f) = std::numeric_limits<typename Field_T::value_type>::quiet_NaN();
         }
      )
   }
}

template< typename Field_T >
void initialiseHomogeneously( std::shared_ptr<StructuredBlockForest> & sbf, const BlockDataID & fieldID,
                              const typename Field_T::value_type & value ) {

   for( auto blockIt = sbf->begin(); blockIt != sbf->end(); ++blockIt ) {

      auto field = blockIt->getData<Field_T>( fieldID );

      WALBERLA_FOR_ALL_CELLS_XYZ(
         field,
         for( uint_t f = 0; f < field->fSize(); ++f ) {
            field->get(x,y,z,f) = value;
         }
      )
   }
}

template< typename Field_T >
void initialiseCheckerboard( std::shared_ptr<StructuredBlockForest> & sbf, const BlockDataID & fieldID,
                             const typename Field_T::value_type & value1, const typename Field_T::value_type & value2 ) {

   for( auto it = sbf->begin(); it != sbf->end(); ++it ) {

      auto field = it->getData<Field_T>( fieldID );

      WALBERLA_FOR_ALL_CELLS_XYZ(
         field,
         typename Field_T::value_type value;
            ((x+y+z) % 2 == 0) ? value = value1 : value = value2;

            for( uint_t f = 0; f < field->fSize(); ++f ) {
               field->get(x,y,z,f) = value * real_t(f+1);
            }
      )

   }
}

/////////////
// CHECKER //
/////////////

template< typename CommunicationStencil_T, typename Field_T >
void checkField( const std::shared_ptr<StructuredBlockForest> & sbf, const BlockDataID & fieldID ) {

   for( auto blockIt = sbf->begin(); blockIt != sbf->end(); ++blockIt ) {

      Block * block = dynamic_cast<Block*>(blockIt.get());

      // get fields
      auto * field = blockIt->getData<Field_T>( fieldID );

      for( auto dir = CommunicationStencil_T::beginNoCenter(); dir != CommunicationStencil_T::end(); ++dir ) {

         // get outer-most interior cells
         CellInterval ci; field->getSliceBeforeGhostLayer( *dir, ci );

         // transform to global cell intervals
         sbf->transformBlockLocalToGlobalCellInterval(ci, *blockIt);

         const auto neighborIdx = blockforest::getBlockNeighborhoodSectionIndex( *dir );

         Vector3<cell_idx_t> coarseShift {
            (stencil::cx[*dir] == 0) ? -1 : 0,
            (stencil::cy[*dir] == 0) ? -1 : 0,
            (CommunicationStencil_T::D == 3) ? ((stencil::cz[*dir] == 0) ? -1 : 0) : 0
         };

         auto neighbourhoodSection = block->getNeighborhoodSection(neighborIdx);
         for(auto nBlock : neighbourhoodSection) {

            auto neighbour    = sbf->getBlock(nBlock->getId());
            auto* neighbourField = neighbour->template getData< Field_T >(fieldID);

            for (auto cell = ci.begin(); cell != ci.end(); ++cell) {

               // get physical coordinate
               auto cellCenter = sbf->getCellCenter(*cell, sbf->getLevel(*blockIt));

               auto neighbourCell = sbf->getBlockLocalCell(*neighbour, cellCenter);
               auto localCell     = sbf->getBlockLocalCell(*blockIt, cellCenter);

               // check equal level communication -> must have the same values
               if (block->neighborhoodSectionHasEquallySizedBlock(neighborIdx)) {

                  for(uint_t f = 0; f < Field_T::F_SIZE; ++ f) {
                     auto fieldValue    = field->get(localCell, f);
                     auto neighborValue = neighbourField->get(neighbourCell, f);
                     WALBERLA_ASSERT_FLOAT_EQUAL(fieldValue, neighborValue,
                                                 "Equal level communication failed.")
                  }

               }
               // check fine-to-coarse communication
               else if (block->neighborhoodSectionHasLargerBlock(neighborIdx)) {

                  Vector3<cell_idx_t> offset {
                     (localCell.x() % 2 == 0) ? +1 : -1,
                     (localCell.y() % 2 == 0) ? +1 : -1,
                     (localCell.z() % 2 == 0) ? +1 : -1
                  };

                  for(uint_t f = 0; f < Field_T::F_SIZE; ++ f) {

                     auto fieldValue  = field->get(localCell[0]          , localCell[1]          , localCell[2], f);
                     fieldValue += field->get(localCell[0]+offset[0], localCell[1]          , localCell[2], f);
                     fieldValue += field->get(localCell[0]          , localCell[1]+offset[1], localCell[2], f);
                     fieldValue += field->get(localCell[0]+offset[0], localCell[1]+offset[1], localCell[2], f);

                     if(CommunicationStencil_T::D == 3) {
                        fieldValue += field->get(localCell[0]          , localCell[1]          , localCell[2]+offset[2], f);
                        fieldValue += field->get(localCell[0]+offset[0], localCell[1]          , localCell[2]+offset[2], f);
                        fieldValue += field->get(localCell[0]          , localCell[1]+offset[1], localCell[2]+offset[2], f);
                        fieldValue += field->get(localCell[0]+offset[0], localCell[1]+offset[1], localCell[2]+offset[2], f);
                     }

                     (CommunicationStencil_T::D == 3) ? fieldValue /= real_t(8) : fieldValue /= real_t(4);

                     auto neighborValue = neighbourField->get(neighbourCell, f);
                     WALBERLA_ASSERT_FLOAT_EQUAL(fieldValue, neighborValue,
                                                 "Fine-to-coarse communication failed for scalar field.")
                  }

               }
               // check coarse-to-fine communication
               else if (block->neighborhoodSectionHasSmallerBlocks(neighborIdx)) {

                  // assure to not check outer most ghost layers of neighbour that are not affected by communication
                  CellInterval neighbourGhostSlice; neighbourField->getGhostRegion( *dir, neighbourGhostSlice, 2 );
                  if(!neighbourGhostSlice.contains(neighbourCell))
                     continue;

                  for(uint_t f = 0; f < Field_T::F_SIZE; ++ f) {
                     auto fieldValue  = field->get(localCell, f);

                     for(cell_idx_t i = coarseShift[0]; i <= 0; ++i) {
                        for(cell_idx_t j = coarseShift[1]; j <= 0; ++j) {
                           for(cell_idx_t k = coarseShift[2]; k <= 0; ++k) {
                              auto neighbourValue = neighbourField->get(neighbourCell[0]+i, neighbourCell[1]+j, neighbourCell[2]+k, f);
                              WALBERLA_ASSERT_FLOAT_EQUAL(fieldValue, neighbourValue,
                                                          "Coarse-to-fine communication failed for global cell <" << cell->x() << ", "
                                                                                                                  << cell->y() << ", "
                                                                                                                  << cell->z() << ">.")
                           }
                        }
                     }
                  }
               }
               else {
                  WALBERLA_ABORT("Something somewhere went terribly wrong!")
               }
            }
         }

      }

   }

}

//////////
// MAIN //
//////////

int main( int argc, char ** argv )
{
   debug::enterTestMode();

   mpi::Environment env( argc, argv );

   logging::Logging::printHeaderOnStream();

   const uint_t levels = uint_t(4);

   const uint_t xBlocks = uint_t(4);
   const uint_t yBlocks = uint_t(4);
   const uint_t zBlocks = uint_t(4);

   const uint_t xCells = uint_t(10);
   const uint_t yCells = uint_t(10);
   const uint_t zCells = uint_t(10);

   auto blocks = createBlockStructure( levels, xBlocks, yBlocks, zBlocks, xCells, yCells, zCells);

   // create fields

   auto scalarFieldID = field::addToStorage<ScalarField_T>(blocks, "scalar field", real_t(0), field::fzyx, FieldGhostLayers);
   auto vectorFieldID = field::addToStorage<VectorField_T>(blocks, "vector field", Vector3<real_t>(0), field::fzyx, FieldGhostLayers);
   auto multiComponentFieldID = field::addToStorage<MultiComponentField_T>(blocks, "multicomponent field", real_t(0), field::fzyx, FieldGhostLayers);

   // create communication scheme

   blockforest::communication::NonUniformBufferedScheme< CommunicationStencil_T > fieldCommunication(blocks);
   fieldCommunication.addPackInfo( std::make_shared<field::refinement::PackInfo<ScalarField_T, CommunicationStencil_T>>( scalarFieldID ) );
   fieldCommunication.addPackInfo( std::make_shared<field::refinement::PackInfo<VectorField_T , CommunicationStencil_T>>( vectorFieldID ) );
   fieldCommunication.addPackInfo( std::make_shared<field::refinement::PackInfo<MultiComponentField_T , CommunicationStencil_T>>( multiComponentFieldID ) );

   ///////////////////////////////
   /// TEST HOMOGENEOUS FIELDS ///
   ///////////////////////////////

   // clear fields
   clearField<ScalarField_T>(blocks, scalarFieldID);
   clearField<VectorField_T>(blocks, vectorFieldID);
   clearField<MultiComponentField_T>(blocks, multiComponentFieldID);

   // initialise fields
   const real_t homogeneousValue{2};
   initialiseHomogeneously<ScalarField_T>(blocks, scalarFieldID, homogeneousValue);
   initialiseHomogeneously<VectorField_T>(blocks, vectorFieldID, Vector3<real_t>(homogeneousValue));
   initialiseHomogeneously<MultiComponentField_T>(blocks, multiComponentFieldID, homogeneousValue);

   // communicate
   fieldCommunication();

   // check ghost layers
   checkField<CommunicationStencil_T, ScalarField_T>(blocks, scalarFieldID);
   checkField<CommunicationStencil_T, VectorField_T>(blocks, vectorFieldID);
   checkField<CommunicationStencil_T, MultiComponentField_T>(blocks, multiComponentFieldID);

   ////////////////////////////////
   /// TEST CHECKERBOARD FIELDS ///
   ////////////////////////////////

   // clear fields
   clearField<ScalarField_T>(blocks, scalarFieldID);
   clearField<VectorField_T>(blocks, vectorFieldID);
   clearField<MultiComponentField_T>(blocks, multiComponentFieldID);

   // initialise fields
   const real_t lowerValue{0};
   const real_t upperValue{2};
   initialiseCheckerboard<ScalarField_T>(blocks, scalarFieldID, lowerValue, upperValue);
   initialiseCheckerboard<VectorField_T>(blocks, vectorFieldID, Vector3<real_t>(lowerValue), Vector3<real_t>(upperValue));
   initialiseCheckerboard<MultiComponentField_T>(blocks, multiComponentFieldID, lowerValue, upperValue);

   // communicate
   fieldCommunication();

   // check ghost layers
   checkField<CommunicationStencil_T, ScalarField_T>(blocks, scalarFieldID);
   checkField<CommunicationStencil_T, VectorField_T>(blocks, vectorFieldID);
   checkField<CommunicationStencil_T, MultiComponentField_T>(blocks, multiComponentFieldID);

   logging::Logging::printFooterOnStream();

   return EXIT_SUCCESS;
}

} // namespace nonuniform_buffered_scheme_test

int main( int argc, char ** argv ) {
   return nonuniform_buffered_scheme_test::main( argc, argv );
}
