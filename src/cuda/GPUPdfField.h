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
//! \file GPUPdfField.h
//! \ingroup cuda
//! \author Markus Holzer <markus.holzer@fau.de>
//
//======================================================================================================================

#pragma once

#include "GPUField.h"

namespace walberla
{
namespace cuda
{
template< typename LatticeModel_T >
class GPUPdfField : public GPUField< real_t >
{
 public:
   //** Type Definitions  **********************************************************************************************
   /*! \name Type Definitions */
   //@{
   typedef LatticeModel_T LatticeModel;
   typedef typename LatticeModel_T::Stencil Stencil;
   //@}
   //*******************************************************************************************************************
   GPUPdfField(uint_t _xSize, uint_t _ySize, uint_t _zSize, uint_t _fSize, const LatticeModel_T& _latticeModel,
               uint_t _nrOfGhostLayers, const Layout& _layout = zyxf, bool usePitchedMem = true);

   virtual ~GPUPdfField() = default;

   const LatticeModel_T& latticeModel() const { return latticeModel_; }
   LatticeModel_T& latticeModel() { return latticeModel_; }

   void resetLatticeModel(const LatticeModel_T& lm) { latticeModel_ = lm; }

   GPUPdfField<LatticeModel_T> * cloneUninitialized() const;

 protected:
   LatticeModel_T latticeModel_;
};

template< typename LatticeModel_T >
GPUPdfField< LatticeModel_T >::GPUPdfField(uint_t _xSize, uint_t _ySize, uint_t _zSize, uint_t _fSize,
                                           const LatticeModel_T& _latticeModel, uint_t _nrOfGhostLayers,
                                           const Layout& _layout, bool usePitchedMem):
     GPUField(_xSize, _ySize, _zSize, _fSize, _nrOfGhostLayers, _layout, usePitchedMem), latticeModel_(_latticeModel)
{

}

template< typename LatticeModel_T >
GPUPdfField<LatticeModel_T> * GPUPdfField<LatticeModel_T>::cloneUninitialized() const
{
   GPUPdfField<LatticeModel_T> * res = new GPUPdfField<LatticeModel_T>( xSize(), ySize(), zSize(), fSize(), latticeModel_,
                                               nrOfGhostLayers(), layout(), isPitchedMem() );

   WALBERLA_ASSERT( hasSameAllocSize( *res ) );
   WALBERLA_ASSERT( hasSameSize( *res ) );
   WALBERLA_ASSERT( layout() == res->layout() );
   WALBERLA_ASSERT( isPitchedMem() == res->isPitchedMem() );
   return res;
}
} // namespace cuda
} // namespace walberla