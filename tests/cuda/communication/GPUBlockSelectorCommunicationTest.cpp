//========================================================================================================================
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
//! \file GPUBlockSelectorCommunicationTest.cpp
//! \ingroup cuda
//! \author Helen Schottenhamml <helen.schottenhamml@fau.de>
//! \brief Short communication test for the usage of block selectors in UniformGPUScheme.
//
//========================================================================================================================

#include <blockforest/Initialization.h>
#include <blockforest/communication/UniformBufferedScheme.h>
#include <core/DataTypes.h>
#include <core/debug/TestSubsystem.h>
#include <core/math/Random.h>
#include <core/mpi/Environment.h>
#include <cuda/AddGPUFieldToStorage.h>
#include <cuda/ErrorChecking.h>
#include <cuda/FieldCopy.h>
#include <cuda/GPUField.h>
#include <cuda/communication/GPUPackInfo.h>
#include <cuda/communication/UniformGPUScheme.h>
#include <cuda_runtime.h>
#include <domain_decomposition/BlockDataID.h>
#include <field/AddToStorage.h>
#include <field/GhostLayerField.h>
#include <stencil/D3Q27.h>
#include <stencil/Directions.h>
#include <stencil/Iterator.h>
#include <vector>

namespace walberla
{
using Type_T = int;

using Stencil_T        = stencil::D3Q27;
using ScalarField_T    = field::GhostLayerField< Type_T, 1 >;
using GPUScalarField_T = cuda::GPUField< Type_T >;
using CommSchemeType   = blockforest::communication::UniformBufferedScheme< StencilType >;
using GPUPackInfoType  = cuda::communication::GPUPackInfo< GPUFieldType >;

void initScalarField(std::shared_ptr< StructuredBlockStorage >& blocks, const BlockDataID& fieldID)
{
   for (auto& block : *blocks)
   {
      Type_T val;
      if (blocks->atDomainXMinBorder(block)) { val = Type_T(0); }
      else if (blocks->atDomainXMaxBorder(block))
      {
         val = Type_T(2);
      }
      else
      {
         val = Type_T(1);
      }

      auto* field = block.getData< ScalarField_T >(fieldID);
      WALBERLA_ASSERT_NOT_NULLPTR(field)

      const auto cells = field->xyzSizeWithGhostLayer();

      for (auto cell : cells)
      {
         field->get(cell) = val;
      }
   }
}

std::shared_ptr< StructuredBlockForest > createUniformBlockGrid(
   const AABB& domainAABB, const uint_t numberOfXBlocks, const uint_t numberOfYBlocks, const uint_t numberOfZBlocks,
   const uint_t numberOfXCellsPerBlock, const uint_t numberOfYCellsPerBlock, const uint_t numberOfZCellsPerBlock,
   const uint_t maxBlocksPerProcess /*= 0*/, const bool includeMetis /*= true*/, const bool forceMetis /*= false*/,
   const bool xPeriodic /*= false*/, const bool yPeriodic /*= false*/, const bool zPeriodic /*= false*/,
   const bool keepGlobalBlockInformation /*= false*/)
{
   // initialize SetupBlockForest = determine domain decomposition

   SetupBlockForest sforest;

   sforest.addWorkloadMemorySUIDAssignmentFunction(uniformWorkloadAndMemoryAssignment);

   sforest.init(domainAABB, numberOfXBlocks, numberOfYBlocks, numberOfZBlocks, xPeriodic, yPeriodic, zPeriodic);

   // calculate process distribution

   const memory_t memoryLimit = (maxBlocksPerProcess == 0) ? numeric_cast< memory_t >(sforest.getNumberOfBlocks()) :
                                                             numeric_cast< memory_t >(maxBlocksPerProcess);

   GlobalLoadBalancing::MetisConfiguration< SetupBlock > metisConfig(
      includeMetis, forceMetis,
      std::bind(cellWeightedCommunicationCost, std::placeholders::_1, std::placeholders::_2, numberOfXCellsPerBlock,
                numberOfYCellsPerBlock, numberOfZCellsPerBlock));

   sforest.calculateProcessDistribution_Default(uint_c(MPIManager::instance()->numProcesses()), memoryLimit, "hilbert",
                                                10, false, metisConfig);

   if (!MPIManager::instance()->rankValid()) MPIManager::instance()->useWorldComm();

   // create StructuredBlockForest (encapsulates a newly created BlockForest)

   auto bf =
      std::make_shared< BlockForest >(uint_c(MPIManager::instance()->rank()), sforest, keepGlobalBlockInformation);

   auto sbf = std::make_shared< StructuredBlockForest >(bf, numberOfXCellsPerBlock, numberOfYCellsPerBlock,
                                                        numberOfZCellsPerBlock);
   sbf->createCellBoundingBoxes();

   return sbf;
}

int main(int argc, char** argv)
{
   debug::enterTestMode();
   walberla::Environment walberlaEnv(argc, argv);

   const Vector3< uint_t > cells = Vector3< uint_t >(2, 2, 2);

   const Set< SUID > requiredBlockSelectors("filled");
   const Set< SUID > incompatibleBlockSelectors("empty");

   auto blocks = blockforest::createUniformBlockGrid(3, 1, 1, cells[0], cells[1], cells[2], 1, false, true, true, true);

   BlockDataID fieldID = field::addToStorage< ScalarField_T >(blocks, "scalar", Type_T(0), field::fzyx, uint_t(1));
   initScalarField(blocks, fieldID);

   BlockDataID gpuFieldID = cuda::addGPUFieldToStorage< GPUScalarField_T >(blocks, fieldID, "GPU scalar");

   // Setup communication schemes for synchronous GPUPackInfo
   cuda::communication::UniformGPUScheme< Stencil_T > communication(blocks, );
   communication.addPackInfo(std::make_shared< cuda::communication::GPUPackInfo< GPUScalarField_T > >(gpuFieldID));

   // Perform one communication step for each scheme
   syncCommScheme();
   asyncCommScheme();

   // Check results
   FieldType syncFieldCpu(cells[0], cells[1], cells[2], 1, fieldLayouts[fieldLayoutIndex],
                          make_shared< cuda::HostFieldAllocator< DataType > >());
   FieldType asyncFieldCpu(cells[0], cells[1], cells[2], 1, fieldLayouts[fieldLayoutIndex],
                           make_shared< cuda::HostFieldAllocator< DataType > >());

   for (auto block = blocks->begin(); block != blocks->end(); ++block)
   {
      auto syncGPUFieldPtr = block->getData< GPUFieldType >(syncGPUFieldId);
      cuda::fieldCpy(syncFieldCpu, *syncGPUFieldPtr);

      auto asyncGPUFieldPtr = block->getData< GPUFieldType >(asyncGPUFieldId);
      cuda::fieldCpy(asyncFieldCpu, *asyncGPUFieldPtr);

      for (auto syncIt = syncFieldCpu.beginWithGhostLayerXYZ(), asyncIt = asyncFieldCpu.beginWithGhostLayerXYZ();
           syncIt != syncFieldCpu.end(); ++syncIt, ++asyncIt)
         WALBERLA_CHECK_EQUAL(*syncIt, *asyncIt);
   }

   return EXIT_SUCCESS;
}

} // namespace walberla

int main(int argc, char** argv) { return walberla::main(argc, argv); }
