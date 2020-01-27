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
//! \file SimpleBB.h
//! \ingroup lbm_mesapd_coupling
//! \author Christoph Rettinger <christoph.rettinger@fau.de>
//
//======================================================================================================================

#pragma once

#include "boundary/Boundary.h"

#include "core/DataTypes.h"
#include "core/cell/CellInterval.h"
#include "core/config/Config.h"
#include "core/debug/Debug.h"
#include "core/logging/all.h"

#include "field/FlagField.h"

#include "lbm/field/PdfField.h"

#include "lbm_mesapd_coupling/DataTypes.h"
#include "lbm_mesapd_coupling/utility/ParticleFunctions.h"

#include "mesa_pd/common/ParticleFunctions.h"
#include "mesa_pd/data/IAccessor.h"

#include "stencil/Directions.h"

namespace walberla {
namespace lbm_mesapd_coupling {

//**************************************************************************************************************************************
/*!
*   \brief Bounce back boundary handling for moving particles
*
*   This boundary condition implements the bounce back scheme to model a no-slip boundary condition for moving particles
*
*/
//**************************************************************************************************************************************

template< typename LatticeModel_T, typename FlagField_T, typename ParticleAccessor_T >
class SimpleBB : public Boundary< typename FlagField_T::flag_t >
{
   using PDFField_T = lbm::PdfField< LatticeModel_T >;
   using Stencil_T = typename LatticeModel_T::Stencil;
   using flag_t = typename FlagField_T::flag_t;

   static_assert(std::is_base_of<mesa_pd::data::IAccessor, ParticleAccessor_T>::value, "Provide a valid accessor as template");

public:

   static shared_ptr<BoundaryConfiguration> createConfiguration( const Config::BlockHandle& )
   {
      WALBERLA_ABORT( "A SimpleBB boundary cannot be created from a config file" );
      return make_shared<BoundaryConfiguration>();
   }

   inline SimpleBB( const BoundaryUID & boundaryUID, const FlagUID & uid, PDFField_T * const pdfField, const FlagField_T * const flagField,
                    ParticleField_T * const particleField, const shared_ptr<ParticleAccessor_T>& ac,
                    const flag_t domain, const StructuredBlockStorage & blockStorage, const IBlock & block );

   void pushFlags( std::vector< FlagUID >& uids ) const { uids.push_back( uid_ ); }

   void beforeBoundaryTreatment() const {}
   void  afterBoundaryTreatment() const {}

   template< typename Buffer_T >
   void packCell( Buffer_T &, const cell_idx_t, const cell_idx_t, const cell_idx_t ) const {}

   template< typename Buffer_T >
   void registerCell( Buffer_T &, const flag_t, const cell_idx_t, const cell_idx_t, const cell_idx_t ) {}

   void registerCell( const flag_t, const cell_idx_t, const cell_idx_t, const cell_idx_t, const BoundaryConfiguration& ) {}
   void registerCells( const flag_t, const CellInterval&, const BoundaryConfiguration& ) {}
   template< typename CellIterator >
   void registerCells( const flag_t, const CellIterator&, const CellIterator&, const BoundaryConfiguration& ) {}

   void unregisterCell( const flag_t, const cell_idx_t, const cell_idx_t, const cell_idx_t ) const {}

   inline void treatDirection( const cell_idx_t  x, const cell_idx_t  y, const cell_idx_t  z, const stencil::Direction dir,
                               const cell_idx_t nx, const cell_idx_t ny, const cell_idx_t nz, const flag_t mask );

private:

   const FlagUID uid_;


   PDFField_T  * const  pdfField_;
   const FlagField_T * const flagField_;
   ParticleField_T * const particleField_;
   shared_ptr<ParticleAccessor_T> ac_;

   flag_t domainMask_;

   const StructuredBlockStorage & blockStorage_;
   const IBlock & block_;

   real_t lengthScalingFactor_;
   real_t forceScalingFactor_;

}; // class SimpleBB


template< typename LatticeModel_T, typename FlagField_T, typename ParticleAccessor_T >
inline SimpleBB< LatticeModel_T, FlagField_T, ParticleAccessor_T >::SimpleBB( const BoundaryUID & boundaryUID, const FlagUID & uid, PDFField_T * const pdfField, const FlagField_T * const flagField,
                                                                              ParticleField_T * const particleField, const shared_ptr<ParticleAccessor_T>& ac,
                                                                              const flag_t domain, const StructuredBlockStorage & blockStorage, const IBlock & block ):
Boundary<flag_t>( boundaryUID ), uid_( uid ), pdfField_( pdfField ), flagField_( flagField ), particleField_( particleField ), ac_( ac ), domainMask_(domain), blockStorage_( blockStorage ), block_( block )
{
   WALBERLA_ASSERT_NOT_NULLPTR( pdfField_ );
   WALBERLA_ASSERT_NOT_NULLPTR( flagField_ );
   WALBERLA_ASSERT_NOT_NULLPTR( particleField_ );
   WALBERLA_ASSERT( flagField_->isRegistered( domainMask_ )  );

   // force scaling factor to account for different dx on this block
   const real_t dxCurrentLevel = blockStorage_.dx( blockStorage_.getLevel(block) );
   lengthScalingFactor_ = dxCurrentLevel;
   forceScalingFactor_ = lengthScalingFactor_ * lengthScalingFactor_;
}

// requires: getVelocityAtPosition (was velFromWF), addForceAtPosition (was addForceAtPos)
template< typename LatticeModel_T, typename FlagField_T, typename ParticleAccessor_T >
#ifndef NDEBUG
inline void SimpleBB< LatticeModel_T, FlagField_T, ParticleAccessor_T >::treatDirection( const cell_idx_t  x, const cell_idx_t  y, const cell_idx_t  z, const stencil::Direction dir,
                                                                                         const cell_idx_t nx, const cell_idx_t ny, const cell_idx_t nz, const flag_t mask )
#else
inline void SimpleBB< LatticeModel_T, FlagField_T, ParticleAccessor_T >::treatDirection( const cell_idx_t  x, const cell_idx_t  y, const cell_idx_t  z, const stencil::Direction dir,
                                                                                         const cell_idx_t nx, const cell_idx_t ny, const cell_idx_t nz, const flag_t /*mask*/ )
#endif
{
   WALBERLA_ASSERT_EQUAL  ( nx, x + cell_idx_t( stencil::cx[ dir ] ) );
   WALBERLA_ASSERT_EQUAL  ( ny, y + cell_idx_t( stencil::cy[ dir ] ) );
   WALBERLA_ASSERT_EQUAL  ( nz, z + cell_idx_t( stencil::cz[ dir ] ) );
   WALBERLA_ASSERT_UNEQUAL( mask & this->mask_, numeric_cast<flag_t>(0) );
   WALBERLA_ASSERT_EQUAL  ( mask & this->mask_, this->mask_ ); // only true if "this->mask_" only contains one single flag, which is the case for the
                                                               // current implementation of this boundary condition
   WALBERLA_ASSERT_UNEQUAL( particleField_->get(nx,ny,nz), ac_->getInvalidUid(), "UID of particle is invalid!" );

   real_t pdf_old = pdfField_->get( x, y, z, Stencil_T::idx[dir] );

   // apply bounce back scheme
   const real_t delta = real_c( 0.5 );
   real_t pdf_new = pdf_old;
   const real_t alpha = real_c(2);

   // get coordinates of fluid cell center
   Cell nearBoundaryCell(x,y,z);
   Vector3< real_t > cellCenter = blockStorage_.getBlockLocalCellCenter(block_, nearBoundaryCell );

   // get vector from fluid cell center to boundary cell center
   Vector3< real_t > direction( lengthScalingFactor_ * real_c( stencil::cx[ dir ] ),
                                lengthScalingFactor_ * real_c( stencil::cy[ dir ] ),
                                lengthScalingFactor_ * real_c( stencil::cz[ dir ] ) );

   //get particle index
   auto particleIdx = ac_->uidToIdx(particleField_->get( nx, ny, nz ));
   WALBERLA_ASSERT_UNEQUAL( particleIdx, ac_->getInvalidIdx(), "Index of particle is invalid!" );

   // assumed boundary position
   const Vector3<real_t> boundaryPosition( cellCenter + delta * direction );

   auto boundaryVelocity = mesa_pd::getVelocityAtWFPoint(particleIdx, *ac_, boundaryPosition );

   // include effect of boundary velocity
   if( LatticeModel_T::compressible )
   {
       const auto density  = pdfField_->getDensity(x,y,z);
       pdf_new -= real_c(3.0) * alpha * density * LatticeModel_T::w[ Stencil_T::idx[dir] ] *
                     ( real_c( stencil::cx[ dir ] ) * boundaryVelocity[0] +
                       real_c( stencil::cy[ dir ] ) * boundaryVelocity[1] +
                       real_c( stencil::cz[ dir ] ) * boundaryVelocity[2] );
   }
   else
   {
       pdf_new -= real_c(3.0) * alpha * LatticeModel_T::w[ Stencil_T::idx[dir] ] *
                     ( real_c( stencil::cx[ dir ] ) * boundaryVelocity[0] +
                       real_c( stencil::cy[ dir ] ) * boundaryVelocity[1] +
                       real_c( stencil::cz[ dir ] ) * boundaryVelocity[2] );
   }

   // carry out the boundary handling
   pdfField_->get( nx, ny, nz, Stencil_T::invDirIdx(dir) ) = pdf_new;

   // check if fluid cell is not inside ghost layer ( = is in inner part), since then no force is allowed to be added
   if( !pdfField_->isInInnerPart( nearBoundaryCell ) )
   {
      return;
   }

   // calculate the force on the obstacle
   // original work (MEM):      Ladd - Numerical simulations of particulate suspensions via a discretized Boltzmann equation. Part 1. Theoretical foundation (1994)
   // improved version (GIMEM): Wen et al. - Galilean invariant fluid-solid interfacial dynamics in lattice Boltzmann simulations (2014)

   if(LatticeModel_T::compressible)
   {
      // this artificially introduces the zero-centering of the PDF values here (see also /lbm/field/PdfField.h)
      // it is important to have correct force values when two or more particles are so close such that there is no fluid cell between
      // as a consequence, some (non-zero) PDF contributions would be missing after summing up the force contributions
      // those would need to be added artificially, see e.g. Ernst, Dietzel, Sommerfeld - A lattice Boltzmann method for simulating transport and agglomeration of resolved particles, Acta Mech, 2013
      // instead, we use the trick there that we just require the deviations from the equilibrium to get the correct force as it is already used for the incompressible case
      pdf_old -= LatticeModel_T::w[ Stencil_T::idx[dir] ];
      pdf_new -= LatticeModel_T::w[ Stencil_T::idx[dir] ];
   }

   // MEM: F = pdf_old + pdf_new - common
   const real_t forceMEM = pdf_old + pdf_new;

   // correction from Wen
   const real_t correction = pdf_old - pdf_new;


   // force consists of the MEM part and the galilean invariance correction including the boundary velocity
   Vector3<real_t> force( real_c( stencil::cx[dir] ) * forceMEM - correction * boundaryVelocity[0],
                          real_c( stencil::cy[dir] ) * forceMEM - correction * boundaryVelocity[1],
                          real_c( stencil::cz[dir] ) * forceMEM - correction * boundaryVelocity[2] );


   force *= forceScalingFactor_;

   // add the force onto the particle at the obstacle boundary
   lbm_mesapd_coupling::addHydrodynamicForceAtWFPosAtomic( particleIdx, *ac_, force, boundaryPosition );

}

} // namespace lbm_mesapd_coupling
} // namespace walberla
