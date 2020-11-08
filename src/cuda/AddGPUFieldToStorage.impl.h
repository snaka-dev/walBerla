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
//! \file AddGPUFieldToStorage.impl.h
//! \ingroup cuda
//! \author Martin Bauer <martin.bauer@fau.de>
//
//======================================================================================================================

#pragma once

#include "cuda/FieldCopy.h"
#include "cuda/GPUPdfField.h"
#include "cuda/GPUBlockDataHandling.h"
#include "lbm/field/PdfField.h"

namespace walberla
{
namespace cuda
{
namespace internal
{
template< typename GPUField_T >
GPUField_T* createGPUField(const IBlock* const block, const StructuredBlockStorage* const bs, uint_t ghostLayers,
                           uint_t fSize, const field::Layout& layout, bool usePitchedMem)
{
   return new GPUField_T(bs->getNumberOfXCells(*block), bs->getNumberOfYCells(*block), bs->getNumberOfZCells(*block),
                         fSize, ghostLayers, layout, usePitchedMem);
}

template< typename Field_T >
GPUField< typename Field_T::value_type >* createGPUFieldFromCPUField(const IBlock* const block,
                                                                     const StructuredBlockStorage* const,
                                                                     ConstBlockDataID cpuFieldID, bool usePitchedMem)
{
   typedef GPUField< typename Field_T::value_type > GPUField_T;

   const Field_T* f = block->getData< Field_T >(cpuFieldID);
   auto gpuField =
      new GPUField_T(f->xSize(), f->ySize(), f->zSize(), f->fSize(), f->nrOfGhostLayers(), f->layout(), usePitchedMem);

   cuda::fieldCpy(*gpuField, *f);

   return gpuField;
}

template<typename PdfField_T, typename LatticeModel_T >
class GPUPdfFieldHandling
   : public cuda::GPUBlockDataHandling< GPUPdfField<LatticeModel_T>, LatticeModel_T::Stencil::D == 2 >
{
 public:
   typedef GPUPdfField< LatticeModel_T > GPUPdfField_T;
   // typedef field::BlockDataHandling< GPUPdfField_T, LatticeModel_T::Stencil::D == 2 > Base_T;

   GPUPdfFieldHandling(const weak_ptr< StructuredBlockStorage >& blocks, ConstBlockDataID cpu_fieldID,
                       const LatticeModel_T& latticeModel, bool usePitchedMem)
      : blocks_(blocks), cpu_fieldID_(cpu_fieldID), latticeModel_(latticeModel), usePitchedMem_(usePitchedMem)
   {}

   inline void serialize(IBlock* const block, const BlockDataID& id, mpi::SendBuffer& buffer)
   {
      packLatticeModel(block, id, buffer);
   }

   void serializeCoarseToFine(Block* const block, const BlockDataID& id, mpi::SendBuffer& buffer, const uint_t child)
   {
      packLatticeModel(block, id, buffer);
   }

   void serializeFineToCoarse(Block* const block, const BlockDataID& id, mpi::SendBuffer& buffer)
   {
      packLatticeModel(block, id, buffer);
   }

   void deserialize(IBlock* const block, const BlockDataID& id, mpi::RecvBuffer& buffer)
   {
      unpackLatticeModel(block, id, buffer);
   }

   void deserializeCoarseToFine(Block* const block, const BlockDataID& id, mpi::RecvBuffer& buffer)
   {
      unpackLatticeModel(block, id, buffer);
   }

   void deserializeFineToCoarse(Block* const block, const BlockDataID& id, mpi::RecvBuffer& buffer, const uint_t child)
   {
      unpackLatticeModel(block, id, buffer);
   }

 protected:
   GPUPdfField_T* allocate(IBlock* const block)
   {
      return allocateDispatch(block, cpu_fieldID_, usePitchedMem_);
   }

   GPUPdfField_T* reallocate(IBlock* const block)
   {
#ifdef NDEBUG
      return allocateDispatch(block, cpu_fieldID_, usePitchedMem_);
#else
      return allocateDispatch(block, cpu_fieldID_, usePitchedMem_);
#endif
   }

 private:
   void packLatticeModel(IBlock* const block, const BlockDataID& id, mpi::SendBuffer& buffer) const
   {
      const GPUPdfField_T * field = block->template getData< GPUPdfField_T >(id);
      WALBERLA_CHECK_NOT_NULLPTR(field);
      buffer << field->latticeModel();
   }

   void unpackLatticeModel(IBlock* const block, const BlockDataID& id, mpi::RecvBuffer& buffer) const
   {
      GPUPdfField_T * field = block->template getData< GPUPdfField_T >(id);
      WALBERLA_CHECK_NOT_NULLPTR(field);

      LatticeModel_T latticeModel = field->latticeModel();
      buffer >> latticeModel;

      auto blocks = blocks_.lock();
      WALBERLA_CHECK_NOT_NULLPTR(blocks);

      latticeModel.configure(*block, *blocks);
      field->resetLatticeModel(latticeModel);
   }

   GPUPdfField_T* allocateDispatch(IBlock* const block, ConstBlockDataID cpu_fieldID, bool usePitchedMem)
   {
      WALBERLA_ASSERT_NOT_NULLPTR( block );

      auto blocks = blocks_.lock();
      WALBERLA_CHECK_NOT_NULLPTR( blocks );
      const PdfField_T * f = block->template getData< PdfField_T >(cpu_fieldID);

      LatticeModel_T latticeModel = f->latticeModel();
      latticeModel_ = latticeModel;

      auto gpuField = new GPUPdfField_T(f->xSize(), f->ySize(), f->zSize(), f->fSize(), latticeModel,
                                        f->nrOfGhostLayers(), f->layout(), usePitchedMem);
      cuda::fieldCpy(*gpuField, *f);

      return gpuField;
   }

   weak_ptr< StructuredBlockStorage > blocks_;

   ConstBlockDataID cpu_fieldID_;
   LatticeModel_T latticeModel_;
   bool usePitchedMem_;

}; // class PdfFieldHandling
} // namespace internal

template< typename GPUField_T >
BlockDataID addGPUFieldToStorage(const shared_ptr< StructuredBlockStorage >& bs, const std::string& identifier,
                                 uint_t fSize, const Layout layout, uint_t nrOfGhostLayers, bool usePitchedMem)
{
   auto func = std::bind(internal::createGPUField< GPUField_T >, std::placeholders::_1, std::placeholders::_2,
                         nrOfGhostLayers, fSize, layout, usePitchedMem);
   return bs->addStructuredBlockData< GPUField_T >(func, identifier);
}

template< typename Field_T >
BlockDataID addGPUFieldToStorage(const shared_ptr< StructuredBlockStorage >& bs, ConstBlockDataID cpuFieldID,
                                 const std::string& identifier, bool usePitchedMem)
{
   auto func = std::bind(internal::createGPUFieldFromCPUField< Field_T >, std::placeholders::_1, std::placeholders::_2,
                         cpuFieldID, usePitchedMem);
   return bs->addStructuredBlockData< GPUField< typename Field_T::value_type > >(func, identifier);
}

template< typename PdfField_T, typename LatticeModel_T, typename BlockStorage_T >
BlockDataID addGPUPdfFieldToStorage(const shared_ptr< BlockStorage_T >& bs, ConstBlockDataID cpu_fieldID,
                                    const LatticeModel_T& _latticeModel, bool usePitchedMem, const std::string& identifier)
{
   return bs->addBlockData(make_shared< internal::GPUPdfFieldHandling<PdfField_T, LatticeModel_T > >(
                                  bs, cpu_fieldID, _latticeModel, usePitchedMem), identifier);
}

} // namespace cuda
} // namespace walberla
