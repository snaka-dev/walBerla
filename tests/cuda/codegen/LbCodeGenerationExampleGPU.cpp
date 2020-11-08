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
//! \file LBCodeGenerationExampleGPU.cpp
//! \author Markus Holzer <markus.holzer@fau.de>
//
//======================================================================================================================

#include "blockforest/all.h"
#include "core/all.h"
#include "domain_decomposition/all.h"
#include "field/all.h"
#include "geometry/all.h"
#include "timeloop/all.h"

#include "cuda/AddGPUFieldToStorage.h"
#include "cuda/DeviceSelectMPI.h"
#include "cuda/communication/UniformGPUScheme.h"


#include "lbm/field/PdfField.h"
#include "lbm/field/AddToStorage.h"
#include "lbm/communication/PdfFieldPackInfo.h"
#include "lbm/gui/Connection.h"
#include "lbm/vtk/VTKOutput.h"

#include "LbCodeGenerationExample_UBB.h"
#include "LbCodeGenerationExample_NoSlip.h"
#include "PackInfo.h"
#include "PDF_Setter.h"
#include "LbCodeGenerationExample_LatticeModel.h"

using namespace walberla;

typedef lbm::LbCodeGenerationExample_LatticeModel LatticeModel_T;
typedef LatticeModel_T::Stencil                   Stencil_T;
typedef LatticeModel_T::CommunicationStencil      CommunicationStencil_T;
typedef lbm::PdfField< LatticeModel_T >           PdfField_T;
typedef pystencils::PackInfo PackInfo_T;

typedef GhostLayerField< real_t, LatticeModel_T::Stencil::D > VectorField_T;
typedef GhostLayerField< real_t, 19 > Test_T;
typedef GhostLayerField< real_t, 1 > ScalarField_T;

typedef walberla::uint8_t    flag_t;
typedef FlagField< flag_t >  FlagField_T;

typedef cuda::GPUField<double> GPUField;

int main( int argc, char ** argv )
{
   walberla::Environment walberlaEnv( argc, argv );
   cuda::selectDeviceBasedOnMpiRank();
   auto blocks = blockforest::createUniformBlockGridFromConfig( walberlaEnv.config() );

   // read parameters
   auto parameters = walberlaEnv.config()->getOneBlock( "Parameters" );

   const real_t          omega           = parameters.getParameter< real_t >         ( "omega",           real_c( 1.4 ) );
   const Vector3<real_t> initialVelocity = parameters.getParameter< Vector3<real_t> >( "initialVelocity", Vector3<real_t>() );
   const uint_t          timesteps       = parameters.getParameter< uint_t >         ( "timesteps",       uint_c( 10 )  );

   const double remainingTimeLoggerFrequency = parameters.getParameter< double >( "remainingTimeLoggerFrequency", 3.0 ); // in seconds

   // create fields
   BlockDataID velFieldId = field::addToStorage<VectorField_T>( blocks, "Velocity", real_t( 0.0 ));
   BlockDataID velFieldIdGPU = cuda::addGPUFieldToStorage<VectorField_T >( blocks, velFieldId, "Velocity field on GPU", true );

   // the generated latticeModel is for GPU usage only. Thus all input data has to be allocated on GPU
   LatticeModel_T latticeModel = LatticeModel_T( velFieldIdGPU, omega );

   // LatticeModel_T latticeModel = LatticeModel_T( velFieldIdGPU, omega );
   // BlockDataID pdfFieldId = field::addToStorage<Test_T>(blocks, "pdf field", real_t(0), field::fzyx);
   BlockDataID pdfFieldId = lbm::addPdfFieldToStorage( blocks, "pdf field", latticeModel, initialVelocity, real_t(1), field::fzyx );
   BlockDataID pdfFieldIdGPU = cuda::addGPUPdfFieldToStorage<PdfField_T, LatticeModel_T>( blocks, pdfFieldId, latticeModel, true, "PDF field on GPU" );


   pystencils::PDF_Setter pdf_setter(pdfFieldIdGPU);

   WALBERLA_LOG_INFO_ON_ROOT("initialization of the distributions")
   for (auto &block : *blocks)
   {
      pdf_setter(&block);
   }

   BlockDataID flagFieldId = field::addFlagFieldToStorage< FlagField_T >( blocks, "flag field" );

   // create and initialize boundary handling
   const FlagUID fluidFlagUID( "Fluid" );


   auto boundariesConfig = walberlaEnv.config()->getOneBlock( "Boundaries" );

   lbm::LbCodeGenerationExample_UBB ubb(blocks, pdfFieldIdGPU);
   lbm::LbCodeGenerationExample_NoSlip noSlip(blocks, pdfFieldIdGPU);

   geometry::initBoundaryHandling<FlagField_T>(*blocks, flagFieldId, boundariesConfig);
   geometry::setNonBoundaryCellsToDomain<FlagField_T>(*blocks, flagFieldId, fluidFlagUID);

   ubb.fillFromFlagField<FlagField_T>( blocks, flagFieldId, FlagUID("UBB"), fluidFlagUID );
   noSlip.fillFromFlagField<FlagField_T>( blocks, flagFieldId, FlagUID("NoSlip"), fluidFlagUID );

   // create time loop
   SweepTimeloop timeloop( blocks->getBlockStorage(), timesteps );

   // create communication for PdfField
   cuda::communication::UniformGPUScheme< Stencil_T > com(blocks, 0);
   com.addPackInfo(make_shared< PackInfo_T >(pdfFieldIdGPU));
   auto communication = std::function< void() >([&]() { com.communicate(nullptr); });

   // add LBM sweep and communication to time loop
   timeloop.add() << BeforeFunction( communication, "communication" )
                  << Sweep( noSlip, "noSlip boundary" );
   timeloop.add() << Sweep( ubb, "ubb boundary" );
   timeloop.add() << Sweep( LatticeModel_T::Sweep( pdfFieldIdGPU ), "LB stream & collide" );

   // log remaining time
   timeloop.addFuncAfterTimeStep( timing::RemainingTimeLogger( timeloop.getNrOfTimeSteps(), remainingTimeLoggerFrequency ), "remaining time logger" );

   // add VTK output to time loop
   uint_t vtkWriteFrequency = parameters.getParameter<uint_t>("VTKwriteFrequency", 0);
   if (vtkWriteFrequency > 0)
   {
      auto vtkOutput = vtk::createVTKOutput_BlockData(*blocks, "vtk", vtkWriteFrequency, 0, false, "vtk_out",
                                                      "simulation_step", false, true, true, false, 0);
      vtkOutput->addBeforeFunction( [&]() {
        cuda::fieldCpy<VectorField_T, GPUField>( blocks, velFieldId, velFieldIdGPU );
      });
      auto Writer = make_shared<field::VTKWriter<VectorField_T>>(velFieldId, "Velocity");
      vtkOutput->addCellDataWriter(Writer);

      timeloop.addFuncBeforeTimeStep(vtk::writeFiles(vtkOutput), "VTK Output");
   }
   // lbm::VTKOutput< LatticeModel_T, FlagField_T >::addToTimeloop( timeloop, blocks, walberlaEnv.config(), pdfFieldId, flagFieldId, fluidFlagUID );

   timeloop.run();

   return EXIT_SUCCESS;
}
