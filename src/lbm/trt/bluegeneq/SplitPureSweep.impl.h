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
//! \file SplitPureSweep.impl.h
//! \ingroup lbm
//! \author Florian Schornbaum <florian.schornbaum@fau.de>
//
//======================================================================================================================



#ifdef __IBMCPP__

#pragma once

#include "lbm/lattice_model/LatticeModelBase.h"
#include "lbm/lattice_model/D3Q19.h"
#include "lbm/sweeps/Streaming.h"
#include "lbm/sweeps/SweepBase.h"

#include <builtins.h>
#include <type_traits>

// OpenMP
#ifdef _OPENMP
#include <omp.h>
#endif

namespace walberla {
namespace lbm {


///////////////////////////////////////////////////////
// Available TRT implementations:                    //
//                                                   //
// There are no generic (D*Q*) versions!             //
//                                                   //
// Optimized D3Q19 implementation:                   //
//                     incompressible | compressible //
//          no forces:       x               x       //
///////////////////////////////////////////////////////


///////////////////////////////
// Specialization for D3Q19: //
// - incompressible          //
// - no additional forces    //
///////////////////////////////

template< typename LatticeModel_T >
class SplitPureSweep< LatticeModel_T, typename std::enable_if< std::is_same< typename LatticeModel_T::CollisionModel::tag, collision_model::TRT_tag >::value &&
                                                               std::is_same< typename LatticeModel_T::Stencil, stencil::D3Q19 >::value &&
                                                               ! LatticeModel_T::compressible &&
                                                               std::is_same< typename LatticeModel_T::ForceModel::tag, force_model::None_tag >::value
                                                               >::type > :
   public SweepBase< LatticeModel_T >
{
public:

   static_assert( (std::is_same< typename LatticeModel_T::CollisionModel::tag, collision_model::TRT_tag >::value), "Only works with TRT!" );
   static_assert( (std::is_same< typename LatticeModel_T::Stencil, stencil::D3Q19 >::value),                       "Only works with D3Q19!" );
   static_assert( LatticeModel_T::compressible == false,                                                             "Only works with incompressible models!" );
   static_assert( (std::is_same< typename LatticeModel_T::ForceModel::tag, force_model::None_tag >::value),        "Only works without additional forces!" );
   static_assert( LatticeModel_T::equilibriumAccuracyOrder == 2, "Only works for lattice models that require the equilibrium distribution to be order 2 accurate!" );

   typedef typename SweepBase<LatticeModel_T>::PdfField_T  PdfField_T;
   typedef typename LatticeModel_T::Stencil                Stencil;

   // block has NO dst pdf field
   SplitPureSweep( const BlockDataID & pdfField ) :
      SweepBase<LatticeModel_T>( pdfField ) {}

   // every block has a dedicated dst pdf field
   SplitPureSweep( const BlockDataID & src, const BlockDataID & dst ) :
      SweepBase<LatticeModel_T>( src, dst ) {}

   void operator()( IBlock * const block );

   void stream ( IBlock * const block, const uint_t numberOfGhostLayersToInclude = uint_t(0) );
   void collide( IBlock * const block, const uint_t numberOfGhostLayersToInclude = uint_t(0) );
};

template< typename LatticeModel_T >
void SplitPureSweep< LatticeModel_T, typename std::enable_if< std::is_same< typename LatticeModel_T::CollisionModel::tag, collision_model::TRT_tag >::value &&
                                                              std::is_same< typename LatticeModel_T::Stencil, stencil::D3Q19 >::value &&
                                                              ! LatticeModel_T::compressible >::value &&
                                                              std::is_same< typename LatticeModel_T::ForceModel::tag, force_model::None_tag >::value
                                                              >::type
   >::operator()( IBlock * const block )
{
   PdfField_T * src( NULL );
   PdfField_T * dst( NULL );

   this->getFields( block, src, dst );

   WALBERLA_ASSERT_NOT_NULLPTR( src );
   WALBERLA_ASSERT_NOT_NULLPTR( dst );

   WALBERLA_ASSERT_GREATER_EQUAL( src->nrOfGhostLayers(), 1 );

   // constants used during stream/collide

   const real_t lambda_e =  src->latticeModel().collisionModel().lambda_e();
   const real_t lambda_d =  src->latticeModel().collisionModel().lambda_d();

   // common prefactors for calculating the equilibrium parts
   const real_t t0   = real_t(1.0) / real_t(3.0);                 // 1/3      for C
   const real_t t1x2 = real_t(1.0) / real_t(18.0) * real_t(2.0);  // 1/18 * 2 for N, S, W, E, T, B
   const real_t t2x2 = real_t(1.0) / real_t(36.0) * real_t(2.0);  // 1/36 * 2 else

   const real_t inv2csq2 = real_t(1.0) / ( real_t(2.0) * ( real_t(1.0) / real_t(3.0) ) * ( real_t(1.0) / real_t(3.0) ) ); //speed of sound related factor for equilibrium distribution function
   const real_t fac1     = t1x2 * inv2csq2;
   const real_t fac2     = t2x2 * inv2csq2;

   // relaxation parameter variables
   const real_t lambda_e_scaled = real_t(0.5) * lambda_e; // 0.5 times the usual value ...
   const real_t lambda_d_scaled = real_t(0.5) * lambda_d; // ... due to the way of calculations

   // loop constants

   const cell_idx_t xSize = cell_idx_c( src->xSize() );
   const cell_idx_t ySize = cell_idx_c( src->ySize() );
   const cell_idx_t zSize = cell_idx_c( src->zSize() );

#ifdef _OPENMP
   #pragma omp parallel
   {
#endif
   // temporaries, calculated by the first innermost loop

   real_t* velX = new real_t[ uint_c(xSize) ];
   real_t* velY = new real_t[ uint_c(xSize) ];
   real_t* velZ = new real_t[ uint_c(xSize) ];

   real_t* feq_common = new real_t[ uint_c(xSize) ];

   if( src->layout() == field::fzyx && dst->layout() == field::fzyx )
   {
      #ifdef _OPENMP
      const int izSize = int_c( zSize );
      #pragma omp for schedule(static)
      for( int iz = 0; iz < izSize; ++iz ) {
         cell_idx_t z = cell_idx_c( iz );
      #else
      for( cell_idx_t z = 0; z < zSize; ++z ) {
      #endif
         for( cell_idx_t y = 0; y != ySize; ++y )
         {
            using namespace stencil;

            real_t* pNE = &src->get(-1, y-1, z  , Stencil::idx[NE]);
            real_t* pN  = &src->get(0 , y-1, z  , Stencil::idx[N]);
            real_t* pNW = &src->get(+1, y-1, z  , Stencil::idx[NW]);
            real_t* pW  = &src->get(+1, y  , z  , Stencil::idx[W]);
            real_t* pSW = &src->get(+1, y+1, z  , Stencil::idx[SW]);
            real_t* pS  = &src->get(0 , y+1, z  , Stencil::idx[S]);
            real_t* pSE = &src->get(-1, y+1, z  , Stencil::idx[SE]);
            real_t* pE  = &src->get(-1, y  , z  , Stencil::idx[E]);
            real_t* pT  = &src->get(0 , y  , z-1, Stencil::idx[T]);
            real_t* pTE = &src->get(-1, y  , z-1, Stencil::idx[TE]);
            real_t* pTN = &src->get(0 , y-1, z-1, Stencil::idx[TN]);
            real_t* pTW = &src->get(+1, y  , z-1, Stencil::idx[TW]);
            real_t* pTS = &src->get(0 , y+1, z-1, Stencil::idx[TS]);
            real_t* pB  = &src->get(0 , y  , z+1, Stencil::idx[B]);
            real_t* pBE = &src->get(-1, y  , z+1, Stencil::idx[BE]);
            real_t* pBN = &src->get(0 , y-1, z+1, Stencil::idx[BN]);
            real_t* pBW = &src->get(+1, y  , z+1, Stencil::idx[BW]);
            real_t* pBS = &src->get(0 , y+1, z+1, Stencil::idx[BS]);
            real_t* pC  = &src->get(0 , y  , z  , Stencil::idx[C]);

            real_t* dC = &dst->get(0,y,z,Stencil::idx[C]);

            __alignx( 32, velX );
            __alignx( 32, velY );
            __alignx( 32, velZ );
            __alignx( 32, feq_common  );

            __alignx( 32, pNE );
            __alignx( 32, pN  );
            __alignx( 32, pNW );
            __alignx( 32, pW  );
            __alignx( 32, pSW );
            __alignx( 32, pS  );
            __alignx( 32, pSE );
            __alignx( 32, pE  );
            __alignx( 32, pT  );
            __alignx( 32, pTE );
            __alignx( 32, pTN );
            __alignx( 32, pTW );
            __alignx( 32, pTS );
            __alignx( 32, pB  );
            __alignx( 32, pBE );
            __alignx( 32, pBN );
            __alignx( 32, pBW );
            __alignx( 32, pBS );
            __alignx( 32, pC  );

            __alignx( 32, dC  );

            #pragma disjoint( *velX, *velY, *velZ, *feq_common, *pNE, *pN, *pNW, *pW, *pSW, *pS, *pSE, \
                              *pE, *pT, *pTE, *pTN, *pTW, *pTS, *pB, *pBE, *pBN, *pBW, *pBS, *pC, *dC, \
                              t0, t1x2, t2x2, fac1, fac2, lambda_e_scaled, lambda_d_scaled )

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t velX_trm = pE[x] + pNE[x] + pSE[x] + pTE[x] + pBE[x];
               const real_t velY_trm = pN[x] + pNW[x] + pTN[x] + pBN[x];
               const real_t velZ_trm = pT[x] + pTS[x] + pTW[x];

               const real_t rho = pC[x] + pS[x] + pW[x] + pB[x] + pSW[x] + pBS[x] + pBW[x] + velX_trm + velY_trm + velZ_trm;

               velX[x] = velX_trm - pW[x]  - pNW[x] - pSW[x] - pTW[x] - pBW[x];
               velY[x] = velY_trm + pNE[x] - pS[x]  - pSW[x] - pSE[x] - pTS[x] - pBS[x];
               velZ[x] = velZ_trm + pTN[x] + pTE[x] - pB[x]  - pBN[x] - pBS[x] - pBW[x] - pBE[x];

               feq_common[x] = rho - real_t(1.5) * ( velX[x] * velX[x] + velY[x] * velY[x] + velZ[x] * velZ[x] );

               dC[x] = pC[x] * (real_t(1.0) - lambda_e) + lambda_e * t0 * feq_common[x];
            }

            real_t* dNE = &dst->get(0,y,z,Stencil::idx[NE]);
            real_t* dSW = &dst->get(0,y,z,Stencil::idx[SW]);

            __alignx( 32, dNE );
            __alignx( 32, dSW );

            #pragma disjoint( *dNE, *dSW, \
                              *velX, *velY, *velZ, *feq_common, *pNE, *pN, *pNW, *pW, *pSW, *pS, *pSE, \
                              *pE, *pT, *pTE, *pTN, *pTW, *pTS, *pB, *pBE, *pBN, *pBW, *pBS, *pC, *dC, \
                              t0, t1x2, t2x2, fac1, fac2, lambda_e_scaled, lambda_d_scaled )

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t velXPY = velX[x] + velY[x];
               const real_t  sym_NE_SW = lambda_e_scaled * ( pNE[x] + pSW[x] - fac2 * velXPY * velXPY - t2x2 * feq_common[x] );
               const real_t asym_NE_SW = lambda_d_scaled * ( pNE[x] - pSW[x] - real_t(3.0) * t2x2 * velXPY );

               dNE[x] = pNE[x] - sym_NE_SW - asym_NE_SW;
               dSW[x] = pSW[x] - sym_NE_SW + asym_NE_SW;
            }

            real_t* dSE = &dst->get(0,y,z,Stencil::idx[SE]);
            real_t* dNW = &dst->get(0,y,z,Stencil::idx[NW]);

            __alignx( 32, dSE );
            __alignx( 32, dNW );

            #pragma disjoint( *dSE, *dNW, *dNE, *dSW, \
                              *velX, *velY, *velZ, *feq_common, *pNE, *pN, *pNW, *pW, *pSW, *pS, *pSE, \
                              *pE, *pT, *pTE, *pTN, *pTW, *pTS, *pB, *pBE, *pBN, *pBW, *pBS, *pC, *dC, \
                              t0, t1x2, t2x2, fac1, fac2, lambda_e_scaled, lambda_d_scaled )

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t velXMY = velX[x] - velY[x];
               const real_t  sym_SE_NW = lambda_e_scaled * ( pSE[x] + pNW[x] - fac2 * velXMY * velXMY - t2x2 * feq_common[x] );
               const real_t asym_SE_NW = lambda_d_scaled * ( pSE[x] - pNW[x] - real_t(3.0) * t2x2 * velXMY );

               dSE[x] = pSE[x] - sym_SE_NW - asym_SE_NW;
               dNW[x] = pNW[x] - sym_SE_NW + asym_SE_NW;
            }

            real_t* dTE = &dst->get(0,y,z,Stencil::idx[TE]);
            real_t* dBW = &dst->get(0,y,z,Stencil::idx[BW]);

            __alignx( 32, dTE );
            __alignx( 32, dBW );

            #pragma disjoint( *dTE, *dBW, *dSE, *dNW, *dNE, *dSW, \
                              *velX, *velY, *velZ, *feq_common, *pNE, *pN, *pNW, *pW, *pSW, *pS, *pSE, \
                              *pE, *pT, *pTE, *pTN, *pTW, *pTS, *pB, *pBE, *pBN, *pBW, *pBS, *pC, *dC, \
                              t0, t1x2, t2x2, fac1, fac2, lambda_e_scaled, lambda_d_scaled )

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t velXPZ = velX[x] + velZ[x];
               const real_t  sym_TE_BW = lambda_e_scaled * ( pTE[x] + pBW[x] - fac2 * velXPZ * velXPZ - t2x2 * feq_common[x] );
               const real_t asym_TE_BW = lambda_d_scaled * ( pTE[x] - pBW[x] - real_t(3.0) * t2x2 * velXPZ );

               dTE[x] = pTE[x] - sym_TE_BW - asym_TE_BW;
               dBW[x] = pBW[x] - sym_TE_BW + asym_TE_BW;
            }

            real_t* dBE = &dst->get(0,y,z,Stencil::idx[BE]);
            real_t* dTW = &dst->get(0,y,z,Stencil::idx[TW]);

            __alignx( 32, dBE );
            __alignx( 32, dTW );

            #pragma disjoint( *dBE, *dTW, *dTE, *dBW, *dSE, *dNW, *dNE, *dSW, \
                              *velX, *velY, *velZ, *feq_common, *pNE, *pN, *pNW, *pW, *pSW, *pS, *pSE, \
                              *pE, *pT, *pTE, *pTN, *pTW, *pTS, *pB, *pBE, *pBN, *pBW, *pBS, *pC, *dC, \
                              t0, t1x2, t2x2, fac1, fac2, lambda_e_scaled, lambda_d_scaled )

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t velXMZ = velX[x] - velZ[x];
               const real_t  sym_BE_TW = lambda_e_scaled * ( pBE[x] + pTW[x] - fac2 * velXMZ * velXMZ - t2x2 * feq_common[x] );
               const real_t asym_BE_TW = lambda_d_scaled * ( pBE[x] - pTW[x] - real_t(3.0) * t2x2 * velXMZ );

               dBE[x] = pBE[x] - sym_BE_TW - asym_BE_TW;
               dTW[x] = pTW[x] - sym_BE_TW + asym_BE_TW;
            }

            real_t* dTN = &dst->get(0,y,z,Stencil::idx[TN]);
            real_t* dBS = &dst->get(0,y,z,Stencil::idx[BS]);

            __alignx( 32, dTN );
            __alignx( 32, dBS );

            #pragma disjoint( *dTN, *dBS, *dBE, *dTW, *dTE, *dBW, *dSE, *dNW, *dNE, *dSW, \
                              *velX, *velY, *velZ, *feq_common, *pNE, *pN, *pNW, *pW, *pSW, *pS, *pSE, \
                              *pE, *pT, *pTE, *pTN, *pTW, *pTS, *pB, *pBE, *pBN, *pBW, *pBS, *pC, *dC, \
                              t0, t1x2, t2x2, fac1, fac2, lambda_e_scaled, lambda_d_scaled )

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t velYPZ = velY[x] + velZ[x];
               const real_t  sym_TN_BS = lambda_e_scaled * ( pTN[x] + pBS[x] - fac2 * velYPZ * velYPZ - t2x2 * feq_common[x] );
               const real_t asym_TN_BS = lambda_d_scaled * ( pTN[x] - pBS[x] - real_t(3.0) * t2x2 * velYPZ );

               dTN[x] = pTN[x] - sym_TN_BS - asym_TN_BS;
               dBS[x] = pBS[x] - sym_TN_BS + asym_TN_BS;
            }

            real_t* dBN = &dst->get(0,y,z,Stencil::idx[BN]);
            real_t* dTS = &dst->get(0,y,z,Stencil::idx[TS]);

            __alignx( 32, dBN );
            __alignx( 32, dTS );

            #pragma disjoint( *dBN, *dTS, *dTN, *dBS, *dBE, *dTW, *dTE, *dBW, *dSE, *dNW, *dNE, *dSW, \
                              *velX, *velY, *velZ, *feq_common, *pNE, *pN, *pNW, *pW, *pSW, *pS, *pSE, \
                              *pE, *pT, *pTE, *pTN, *pTW, *pTS, *pB, *pBE, *pBN, *pBW, *pBS, *pC, *dC, \
                              t0, t1x2, t2x2, fac1, fac2, lambda_e_scaled, lambda_d_scaled )

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t velYMZ = velY[x] - velZ[x];
               const real_t  sym_BN_TS = lambda_e_scaled * ( pBN[x] + pTS[x] - fac2 * velYMZ * velYMZ - t2x2 * feq_common[x] );
               const real_t asym_BN_TS = lambda_d_scaled * ( pBN[x] - pTS[x] - real_t(3.0) * t2x2 * velYMZ );

               dBN[x] = pBN[x] - sym_BN_TS - asym_BN_TS;
               dTS[x] = pTS[x] - sym_BN_TS + asym_BN_TS;
            }

            real_t* dN = &dst->get(0,y,z,Stencil::idx[N]);
            real_t* dS = &dst->get(0,y,z,Stencil::idx[S]);

            __alignx( 32, dN );
            __alignx( 32, dS );

            #pragma disjoint( *dN, *dS, *dBN, *dTS, *dTN, *dBS, *dBE, *dTW, *dTE, *dBW, *dSE, *dNW, *dNE, *dSW, \
                              *velX, *velY, *velZ, *feq_common, *pNE, *pN, *pNW, *pW, *pSW, *pS, *pSE, \
                              *pE, *pT, *pTE, *pTN, *pTW, *pTS, *pB, *pBE, *pBN, *pBW, *pBS, *pC, *dC, \
                              t0, t1x2, t2x2, fac1, fac2, lambda_e_scaled, lambda_d_scaled )

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t  sym_N_S = lambda_e_scaled * ( pN[x] + pS[x] - fac1 * velY[x] * velY[x] - t1x2 * feq_common[x] );
               const real_t asym_N_S = lambda_d_scaled * ( pN[x] - pS[x] - real_t(3.0) * t1x2 * velY[x] );

               dN[x] = pN[x] - sym_N_S - asym_N_S;
               dS[x] = pS[x] - sym_N_S + asym_N_S;
            }

            real_t* dE = &dst->get(0,y,z,Stencil::idx[E]);
            real_t* dW = &dst->get(0,y,z,Stencil::idx[W]);

            __alignx( 32, dE );
            __alignx( 32, dW );

            #pragma disjoint( *dE, *dW, *dN, *dS, *dBN, *dTS, *dTN, *dBS, *dBE, *dTW, *dTE, *dBW, *dSE, *dNW, *dNE, *dSW, \
                              *velX, *velY, *velZ, *feq_common, *pNE, *pN, *pNW, *pW, *pSW, *pS, *pSE, \
                              *pE, *pT, *pTE, *pTN, *pTW, *pTS, *pB, *pBE, *pBN, *pBW, *pBS, *pC, *dC, \
                              t0, t1x2, t2x2, fac1, fac2, lambda_e_scaled, lambda_d_scaled )

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t  sym_E_W = lambda_e_scaled * ( pE[x] + pW[x] - fac1 * velX[x] * velX[x] - t1x2 * feq_common[x] );
               const real_t asym_E_W = lambda_d_scaled * ( pE[x] - pW[x] - real_t(3.0) * t1x2 * velX[x] );

               dE[x] = pE[x] - sym_E_W - asym_E_W;
               dW[x] = pW[x] - sym_E_W + asym_E_W;
            }

            real_t* dT = &dst->get(0,y,z,Stencil::idx[T]);
            real_t* dB = &dst->get(0,y,z,Stencil::idx[B]);

            __alignx( 32, dT );
            __alignx( 32, dB );

            #pragma disjoint( *dT, *dB, *dE, *dW, *dN, *dS, *dBN, *dTS, *dTN, *dBS, *dBE, *dTW, *dTE, *dBW, *dSE, *dNW, *dNE, *dSW, \
                              *velX, *velY, *velZ, *feq_common, *pNE, *pN, *pNW, *pW, *pSW, *pS, *pSE, \
                              *pE, *pT, *pTE, *pTN, *pTW, *pTS, *pB, *pBE, *pBN, *pBW, *pBS, *pC, *dC, \
                              t0, t1x2, t2x2, fac1, fac2, lambda_e_scaled, lambda_d_scaled )

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t  sym_T_B = lambda_e_scaled * ( pT[x] + pB[x]  - fac1 * velZ[x] * velZ[x] - t1x2 * feq_common[x] );
               const real_t asym_T_B = lambda_d_scaled * ( pT[x] - pB[x] - real_t(3.0) * t1x2 * velZ[x] );

               dT[x] = pT[x] - sym_T_B - asym_T_B;
               dB[x] = pB[x] - sym_T_B + asym_T_B;
            }
         }
      }
   }
   else // ==> src->layout() == field::zyxf || dst->layout() == field::zyxf
   {
      #ifdef _OPENMP
      const int izSize = int_c( zSize );
      #pragma omp for schedule(static)
      for( int iz = 0; iz < izSize; ++iz ) {
         cell_idx_t z = cell_idx_c( iz );
      #else
      for( cell_idx_t z = 0; z < zSize; ++z ) {
      #endif
         for( cell_idx_t y = 0; y != ySize; ++y )
         {
            using namespace stencil;

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_NE = src->get(x-1, y-1, z  , Stencil::idx[NE]);
               const real_t dd_tmp_N  = src->get(x  , y-1, z  , Stencil::idx[N]);
               const real_t dd_tmp_NW = src->get(x+1, y-1, z  , Stencil::idx[NW]);
               const real_t dd_tmp_W  = src->get(x+1, y  , z  , Stencil::idx[W]);
               const real_t dd_tmp_SW = src->get(x+1, y+1, z  , Stencil::idx[SW]);
               const real_t dd_tmp_S  = src->get(x  , y+1, z  , Stencil::idx[S]);
               const real_t dd_tmp_SE = src->get(x-1, y+1, z  , Stencil::idx[SE]);
               const real_t dd_tmp_E  = src->get(x-1, y  , z  , Stencil::idx[E]);
               const real_t dd_tmp_T  = src->get(x  , y  , z-1, Stencil::idx[T]);
               const real_t dd_tmp_TE = src->get(x-1, y  , z-1, Stencil::idx[TE]);
               const real_t dd_tmp_TN = src->get(x  , y-1, z-1, Stencil::idx[TN]);
               const real_t dd_tmp_TW = src->get(x+1, y  , z-1, Stencil::idx[TW]);
               const real_t dd_tmp_TS = src->get(x  , y+1, z-1, Stencil::idx[TS]);
               const real_t dd_tmp_B  = src->get(x  , y  , z+1, Stencil::idx[B]);
               const real_t dd_tmp_BE = src->get(x-1, y  , z+1, Stencil::idx[BE]);
               const real_t dd_tmp_BN = src->get(x  , y-1, z+1, Stencil::idx[BN]);
               const real_t dd_tmp_BW = src->get(x+1, y  , z+1, Stencil::idx[BW]);
               const real_t dd_tmp_BS = src->get(x  , y+1, z+1, Stencil::idx[BS]);
               const real_t dd_tmp_C  = src->get(x  , y  , z  , Stencil::idx[C]);

               const real_t velX_trm = dd_tmp_E + dd_tmp_NE + dd_tmp_SE + dd_tmp_TE + dd_tmp_BE;
               const real_t velY_trm = dd_tmp_N + dd_tmp_NW + dd_tmp_TN + dd_tmp_BN;
               const real_t velZ_trm = dd_tmp_T + dd_tmp_TS + dd_tmp_TW;

               const real_t rho = dd_tmp_C + dd_tmp_S + dd_tmp_W + dd_tmp_B + dd_tmp_SW + dd_tmp_BS + dd_tmp_BW + velX_trm + velY_trm + velZ_trm;

               velX[x] = velX_trm - dd_tmp_W  - dd_tmp_NW - dd_tmp_SW - dd_tmp_TW - dd_tmp_BW;
               velY[x] = velY_trm + dd_tmp_NE - dd_tmp_S  - dd_tmp_SW - dd_tmp_SE - dd_tmp_TS - dd_tmp_BS;
               velZ[x] = velZ_trm + dd_tmp_TN + dd_tmp_TE - dd_tmp_B  - dd_tmp_BN - dd_tmp_BS - dd_tmp_BW - dd_tmp_BE;

               feq_common[x] = rho - real_t(1.5) * ( velX[x] * velX[x] + velY[x] * velY[x] + velZ[x] * velZ[x] );

               dst->get( x, y, z, Stencil::idx[C] ) = dd_tmp_C * (real_t(1.0) - lambda_e) + lambda_e * t0 * feq_common[x];
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_NE = src->get(x-1, y-1, z, Stencil::idx[NE]);
               const real_t dd_tmp_SW = src->get(x+1, y+1, z, Stencil::idx[SW]);

               const real_t velXPY = velX[x] + velY[x];
               const real_t  sym_NE_SW = lambda_e_scaled * ( dd_tmp_NE + dd_tmp_SW - fac2 * velXPY * velXPY - t2x2 * feq_common[x] );
               const real_t asym_NE_SW = lambda_d_scaled * ( dd_tmp_NE - dd_tmp_SW - real_t(3.0) * t2x2 * velXPY );

               dst->get( x, y, z, Stencil::idx[NE] ) = dd_tmp_NE - sym_NE_SW - asym_NE_SW;
               dst->get( x, y, z, Stencil::idx[SW] ) = dd_tmp_SW - sym_NE_SW + asym_NE_SW;
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_SE = src->get(x-1, y+1, z, Stencil::idx[SE]);
               const real_t dd_tmp_NW = src->get(x+1, y-1, z, Stencil::idx[NW]);

               const real_t velXMY = velX[x] - velY[x];
               const real_t  sym_SE_NW = lambda_e_scaled * ( dd_tmp_SE + dd_tmp_NW - fac2 * velXMY * velXMY - t2x2 * feq_common[x] );
               const real_t asym_SE_NW = lambda_d_scaled * ( dd_tmp_SE - dd_tmp_NW - real_t(3.0) * t2x2 * velXMY );

               dst->get( x, y, z, Stencil::idx[SE] ) = dd_tmp_SE - sym_SE_NW - asym_SE_NW;
               dst->get( x, y, z, Stencil::idx[NW] ) = dd_tmp_NW - sym_SE_NW + asym_SE_NW;
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_TE = src->get(x-1, y, z-1, Stencil::idx[TE]);
               const real_t dd_tmp_BW = src->get(x+1, y, z+1, Stencil::idx[BW]);

               const real_t velXPZ = velX[x] + velZ[x];
               const real_t  sym_TE_BW = lambda_e_scaled * ( dd_tmp_TE + dd_tmp_BW - fac2 * velXPZ * velXPZ - t2x2 * feq_common[x] );
               const real_t asym_TE_BW = lambda_d_scaled * ( dd_tmp_TE - dd_tmp_BW - real_t(3.0) * t2x2 * velXPZ );

               dst->get( x, y, z, Stencil::idx[TE] ) = dd_tmp_TE - sym_TE_BW - asym_TE_BW;
               dst->get( x, y, z, Stencil::idx[BW] ) = dd_tmp_BW - sym_TE_BW + asym_TE_BW;
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_BE = src->get(x-1, y, z+1, Stencil::idx[BE]);
               const real_t dd_tmp_TW = src->get(x+1, y, z-1, Stencil::idx[TW]);

               const real_t velXMZ = velX[x] - velZ[x];
               const real_t  sym_BE_TW = lambda_e_scaled * ( dd_tmp_BE + dd_tmp_TW - fac2 * velXMZ * velXMZ - t2x2 * feq_common[x] );
               const real_t asym_BE_TW = lambda_d_scaled * ( dd_tmp_BE - dd_tmp_TW - real_t(3.0) * t2x2 * velXMZ );

               dst->get( x, y, z, Stencil::idx[BE] ) = dd_tmp_BE - sym_BE_TW - asym_BE_TW;
               dst->get( x, y, z, Stencil::idx[TW] ) = dd_tmp_TW - sym_BE_TW + asym_BE_TW;
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_TN = src->get(x, y-1, z-1, Stencil::idx[TN]);
               const real_t dd_tmp_BS = src->get(x, y+1, z+1, Stencil::idx[BS]);

               const real_t velYPZ = velY[x] + velZ[x];
               const real_t  sym_TN_BS = lambda_e_scaled * ( dd_tmp_TN + dd_tmp_BS - fac2 * velYPZ * velYPZ - t2x2 * feq_common[x] );
               const real_t asym_TN_BS = lambda_d_scaled * ( dd_tmp_TN - dd_tmp_BS - real_t(3.0) * t2x2 * velYPZ );

               dst->get( x, y, z, Stencil::idx[TN] ) = dd_tmp_TN - sym_TN_BS - asym_TN_BS;
               dst->get( x, y, z, Stencil::idx[BS] ) = dd_tmp_BS - sym_TN_BS + asym_TN_BS;
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_BN = src->get(x, y-1, z+1, Stencil::idx[BN]);
               const real_t dd_tmp_TS = src->get(x, y+1, z-1, Stencil::idx[TS]);

               const real_t velYMZ = velY[x] - velZ[x];
               const real_t  sym_BN_TS = lambda_e_scaled * ( dd_tmp_BN + dd_tmp_TS - fac2 * velYMZ * velYMZ - t2x2 * feq_common[x] );
               const real_t asym_BN_TS = lambda_d_scaled * ( dd_tmp_BN - dd_tmp_TS - real_t(3.0) * t2x2 * velYMZ );

               dst->get( x, y, z, Stencil::idx[BN] ) = dd_tmp_BN - sym_BN_TS - asym_BN_TS;
               dst->get( x, y, z, Stencil::idx[TS] ) = dd_tmp_TS - sym_BN_TS + asym_BN_TS;
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_N  = src->get(x, y-1, z, Stencil::idx[N]);
               const real_t dd_tmp_S  = src->get(x, y+1, z, Stencil::idx[S]);

               const real_t  sym_N_S = lambda_e_scaled * ( dd_tmp_N + dd_tmp_S - fac1 * velY[x] * velY[x] - t1x2 * feq_common[x] );
               const real_t asym_N_S = lambda_d_scaled * ( dd_tmp_N - dd_tmp_S - real_t(3.0) * t1x2 * velY[x] );

               dst->get( x, y, z, Stencil::idx[N] ) = dd_tmp_N - sym_N_S - asym_N_S;
               dst->get( x, y, z, Stencil::idx[S] ) = dd_tmp_S - sym_N_S + asym_N_S;
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_E  = src->get(x-1, y, z, Stencil::idx[E]);
               const real_t dd_tmp_W  = src->get(x+1, y, z, Stencil::idx[W]);

               const real_t  sym_E_W = lambda_e_scaled * ( dd_tmp_E + dd_tmp_W - fac1 * velX[x] * velX[x] - t1x2 * feq_common[x] );
               const real_t asym_E_W = lambda_d_scaled * ( dd_tmp_E - dd_tmp_W - real_t(3.0) * t1x2 * velX[x] );

               dst->get( x, y, z, Stencil::idx[E] ) = dd_tmp_E - sym_E_W - asym_E_W;
               dst->get( x, y, z, Stencil::idx[W] ) = dd_tmp_W - sym_E_W + asym_E_W;
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_T  = src->get(x, y, z-1, Stencil::idx[T]);
               const real_t dd_tmp_B  = src->get(x, y, z+1, Stencil::idx[B]);

               const real_t  sym_T_B = lambda_e_scaled * ( dd_tmp_T + dd_tmp_B  - fac1 * velZ[x] * velZ[x] - t1x2 * feq_common[x] );
               const real_t asym_T_B = lambda_d_scaled * ( dd_tmp_T - dd_tmp_B - real_t(3.0) * t1x2 * velZ[x] );

               dst->get( x, y, z, Stencil::idx[T] ) = dd_tmp_T - sym_T_B - asym_T_B;
               dst->get( x, y, z, Stencil::idx[B] ) = dd_tmp_B - sym_T_B + asym_T_B;
            }
         }
      }
   }

   delete[] velX;
   delete[] velY;
   delete[] velZ;
   delete[] feq_common;

#ifdef _OPENMP
   }
#endif

   src->swapDataPointers( dst );
}

template< typename LatticeModel_T >
void SplitPureSweep< LatticeModel_T, typename std::enable_if< std::is_same< typename LatticeModel_T::CollisionModel::tag, collision_model::TRT_tag >::value &&
                                                              std::is_same< typename LatticeModel_T::Stencil, stencil::D3Q19 >::value &&
                                                              ! LatticeModel_T::compressible >::value &&
                                                              std::is_same< typename LatticeModel_T::ForceModel::tag, force_model::None_tag >::value
                                                              >::type
   >::stream( IBlock * const block, const uint_t numberOfGhostLayersToInclude )
{
   PdfField_T * src( NULL );
   PdfField_T * dst( NULL );

   this->getFields( block, src, dst );

   StreamEverything< LatticeModel_T >::execute( src, dst, numberOfGhostLayersToInclude );
}

template< typename LatticeModel_T >
void SplitPureSweep< LatticeModel_T, typename std::enable_if< std::is_same< typename LatticeModel_T::CollisionModel::tag, collision_model::TRT_tag >::value &&
                                                              std::is_same< typename LatticeModel_T::Stencil, stencil::D3Q19 >::value &&
                                                              ! LatticeModel_T::compressible >::value &&
                                                              std::is_same< typename LatticeModel_T::ForceModel::tag, force_model::None_tag >::value
                                                              >::type
#ifdef NDEBUG
   >::collide( IBlock * const block, const uint_t /*numberOfGhostLayersToInclude*/ )
#else
   >::collide( IBlock * const block, const uint_t numberOfGhostLayersToInclude )
#endif
{
   WALBERLA_ASSERT_EQUAL( numberOfGhostLayersToInclude, uint_t(0) ); // the implementation right now doesn't support inclusion of ghost layers in collide step!

   PdfField_T * src = this->getSrcField( block );

   WALBERLA_ASSERT_NOT_NULLPTR( src );

   WALBERLA_ASSERT_GREATER_EQUAL( src->nrOfGhostLayers(), numberOfGhostLayersToInclude );

   // constants used during stream/collide

   const real_t lambda_e =  src->latticeModel().collisionModel().lambda_e();
   const real_t lambda_d =  src->latticeModel().collisionModel().lambda_d();

   // common prefactors for calculating the equilibrium parts
   const real_t t0   = real_t(1.0) / real_t(3.0);                 // 1/3      for C
   const real_t t1x2 = real_t(1.0) / real_t(18.0) * real_t(2.0);  // 1/18 * 2 for N, S, W, E, T, B
   const real_t t2x2 = real_t(1.0) / real_t(36.0) * real_t(2.0);  // 1/36 * 2 else

   const real_t inv2csq2 = real_t(1.0) / ( real_t(2.0) * ( real_t(1.0) / real_t(3.0) ) * ( real_t(1.0) / real_t(3.0) ) ); //speed of sound related factor for equilibrium distribution function
   const real_t fac1     = t1x2 * inv2csq2;
   const real_t fac2     = t2x2 * inv2csq2;

   // relaxation parameter variables
   const real_t lambda_e_scaled = real_t(0.5) * lambda_e; // 0.5 times the usual value ...
   const real_t lambda_d_scaled = real_t(0.5) * lambda_d; // ... due to the way of calculations

   // loop constants

   const cell_idx_t xSize = cell_idx_c( src->xSize() );
   const cell_idx_t ySize = cell_idx_c( src->ySize() );
   const cell_idx_t zSize = cell_idx_c( src->zSize() );

#ifdef _OPENMP
   #pragma omp parallel
   {
#endif
   // temporaries, calculated by the first innermost loop

   real_t* velX = new real_t[ uint_c(xSize) ];
   real_t* velY = new real_t[ uint_c(xSize) ];
   real_t* velZ = new real_t[ uint_c(xSize) ];

   real_t* feq_common = new real_t[ uint_c(xSize) ];

   if( src->layout() == field::fzyx )
   {
      #ifdef _OPENMP
      const int izSize = int_c( zSize );
      #pragma omp for schedule(static)
      for( int iz = 0; iz < izSize; ++iz ) {
         cell_idx_t z = cell_idx_c( iz );
      #else
      for( cell_idx_t z = 0; z < zSize; ++z ) {
      #endif
         for( cell_idx_t y = 0; y != ySize; ++y )
         {
            using namespace stencil;

            real_t* pC  = &src->get( 0, y, z, Stencil::idx[C]);
            real_t* pN  = &src->get( 0, y, z, Stencil::idx[N]);
            real_t* pS  = &src->get( 0, y, z, Stencil::idx[S]);
            real_t* pW  = &src->get( 0, y, z, Stencil::idx[W]);
            real_t* pE  = &src->get( 0, y, z, Stencil::idx[E]);
            real_t* pT  = &src->get( 0, y, z, Stencil::idx[T]);
            real_t* pB  = &src->get( 0, y, z, Stencil::idx[B]);
            real_t* pNW = &src->get( 0, y, z, Stencil::idx[NW]);
            real_t* pNE = &src->get( 0, y, z, Stencil::idx[NE]);
            real_t* pSW = &src->get( 0, y, z, Stencil::idx[SW]);
            real_t* pSE = &src->get( 0, y, z, Stencil::idx[SE]);
            real_t* pTN = &src->get( 0, y, z, Stencil::idx[TN]);
            real_t* pTS = &src->get( 0, y, z, Stencil::idx[TS]);
            real_t* pTW = &src->get( 0, y, z, Stencil::idx[TW]);
            real_t* pTE = &src->get( 0, y, z, Stencil::idx[TE]);
            real_t* pBN = &src->get( 0, y, z, Stencil::idx[BN]);
            real_t* pBS = &src->get( 0, y, z, Stencil::idx[BS]);
            real_t* pBW = &src->get( 0, y, z, Stencil::idx[BW]);
            real_t* pBE = &src->get( 0, y, z, Stencil::idx[BE]);

            __alignx( 32, velX );
            __alignx( 32, velY );
            __alignx( 32, velZ );
            __alignx( 32, feq_common  );

            __alignx( 32, pNE );
            __alignx( 32, pN  );
            __alignx( 32, pNW );
            __alignx( 32, pW  );
            __alignx( 32, pSW );
            __alignx( 32, pS  );
            __alignx( 32, pSE );
            __alignx( 32, pE  );
            __alignx( 32, pT  );
            __alignx( 32, pTE );
            __alignx( 32, pTN );
            __alignx( 32, pTW );
            __alignx( 32, pTS );
            __alignx( 32, pB  );
            __alignx( 32, pBE );
            __alignx( 32, pBN );
            __alignx( 32, pBW );
            __alignx( 32, pBS );
            __alignx( 32, pC  );

            #pragma disjoint( *velX, *velY, *velZ, *feq_common, *pNE, *pN, *pNW, *pW, *pSW, *pS, *pSE, \
                              *pE, *pT, *pTE, *pTN, *pTW, *pTS, *pB, *pBE, *pBN, *pBW, *pBS, *pC, \
                              t0, t1x2, t2x2, fac1, fac2, lambda_e_scaled, lambda_d_scaled )

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t velX_trm = pE[x] + pNE[x] + pSE[x] + pTE[x] + pBE[x];
               const real_t velY_trm = pN[x] + pNW[x] + pTN[x] + pBN[x];
               const real_t velZ_trm = pT[x] + pTS[x] + pTW[x];

               const real_t rho = pC[x] + pS[x] + pW[x] + pB[x] + pSW[x] + pBS[x] + pBW[x] + velX_trm + velY_trm + velZ_trm;

               velX[x] = velX_trm - pW[x]  - pNW[x] - pSW[x] - pTW[x] - pBW[x];
               velY[x] = velY_trm + pNE[x] - pS[x]  - pSW[x] - pSE[x] - pTS[x] - pBS[x];
               velZ[x] = velZ_trm + pTN[x] + pTE[x] - pB[x]  - pBN[x] - pBS[x] - pBW[x] - pBE[x];

               feq_common[x] = rho - real_t(1.5) * ( velX[x] * velX[x] + velY[x] * velY[x] + velZ[x] * velZ[x] );

               pC[x] = pC[x] * (real_t(1.0) - lambda_e) + lambda_e * t0 * feq_common[x];
            }

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t velXPY = velX[x] + velY[x];
               const real_t  sym_NE_SW = lambda_e_scaled * ( pNE[x] + pSW[x] - fac2 * velXPY * velXPY - t2x2 * feq_common[x] );
               const real_t asym_NE_SW = lambda_d_scaled * ( pNE[x] - pSW[x] - real_t(3.0) * t2x2 * velXPY );

               pNE[x] = pNE[x] - sym_NE_SW - asym_NE_SW;
               pSW[x] = pSW[x] - sym_NE_SW + asym_NE_SW;
            }

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t velXMY = velX[x] - velY[x];
               const real_t  sym_SE_NW = lambda_e_scaled * ( pSE[x] + pNW[x] - fac2 * velXMY * velXMY - t2x2 * feq_common[x] );
               const real_t asym_SE_NW = lambda_d_scaled * ( pSE[x] - pNW[x] - real_t(3.0) * t2x2 * velXMY );

               pSE[x] = pSE[x] - sym_SE_NW - asym_SE_NW;
               pNW[x] = pNW[x] - sym_SE_NW + asym_SE_NW;
            }

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t velXPZ = velX[x] + velZ[x];
               const real_t  sym_TE_BW = lambda_e_scaled * ( pTE[x] + pBW[x] - fac2 * velXPZ * velXPZ - t2x2 * feq_common[x] );
               const real_t asym_TE_BW = lambda_d_scaled * ( pTE[x] - pBW[x] - real_t(3.0) * t2x2 * velXPZ );

               pTE[x] = pTE[x] - sym_TE_BW - asym_TE_BW;
               pBW[x] = pBW[x] - sym_TE_BW + asym_TE_BW;
            }

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t velXMZ = velX[x] - velZ[x];
               const real_t  sym_BE_TW = lambda_e_scaled * ( pBE[x] + pTW[x] - fac2 * velXMZ * velXMZ - t2x2 * feq_common[x] );
               const real_t asym_BE_TW = lambda_d_scaled * ( pBE[x] - pTW[x] - real_t(3.0) * t2x2 * velXMZ );

               pBE[x] = pBE[x] - sym_BE_TW - asym_BE_TW;
               pTW[x] = pTW[x] - sym_BE_TW + asym_BE_TW;
            }

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t velYPZ = velY[x] + velZ[x];
               const real_t  sym_TN_BS = lambda_e_scaled * ( pTN[x] + pBS[x] - fac2 * velYPZ * velYPZ - t2x2 * feq_common[x] );
               const real_t asym_TN_BS = lambda_d_scaled * ( pTN[x] - pBS[x] - real_t(3.0) * t2x2 * velYPZ );

               pTN[x] = pTN[x] - sym_TN_BS - asym_TN_BS;
               pBS[x] = pBS[x] - sym_TN_BS + asym_TN_BS;
            }

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t velYMZ = velY[x] - velZ[x];
               const real_t  sym_BN_TS = lambda_e_scaled * ( pBN[x] + pTS[x] - fac2 * velYMZ * velYMZ - t2x2 * feq_common[x] );
               const real_t asym_BN_TS = lambda_d_scaled * ( pBN[x] - pTS[x] - real_t(3.0) * t2x2 * velYMZ );

               pBN[x] = pBN[x] - sym_BN_TS - asym_BN_TS;
               pTS[x] = pTS[x] - sym_BN_TS + asym_BN_TS;
            }

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t  sym_N_S = lambda_e_scaled * ( pN[x] + pS[x] - fac1 * velY[x] * velY[x] - t1x2 * feq_common[x] );
               const real_t asym_N_S = lambda_d_scaled * ( pN[x] - pS[x] - real_t(3.0) * t1x2 * velY[x] );

               pN[x] = pN[x] - sym_N_S - asym_N_S;
               pS[x] = pS[x] - sym_N_S + asym_N_S;
            }

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t  sym_E_W = lambda_e_scaled * ( pE[x] + pW[x] - fac1 * velX[x] * velX[x] - t1x2 * feq_common[x] );
               const real_t asym_E_W = lambda_d_scaled * ( pE[x] - pW[x] - real_t(3.0) * t1x2 * velX[x] );

               pE[x] = pE[x] - sym_E_W - asym_E_W;
               pW[x] = pW[x] - sym_E_W + asym_E_W;
            }

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t  sym_T_B = lambda_e_scaled * ( pT[x] + pB[x]  - fac1 * velZ[x] * velZ[x] - t1x2 * feq_common[x] );
               const real_t asym_T_B = lambda_d_scaled * ( pT[x] - pB[x] - real_t(3.0) * t1x2 * velZ[x] );

               pT[x] = pT[x] - sym_T_B - asym_T_B;
               pB[x] = pB[x] - sym_T_B + asym_T_B;
            }
         }
      }
   }
   else // ==> src->layout() == field::zyxf
   {
      #ifdef _OPENMP
      const int izSize = int_c( zSize );
      #pragma omp for schedule(static)
      for( int iz = 0; iz < izSize; ++iz ) {
         cell_idx_t z = cell_idx_c( iz );
      #else
      for( cell_idx_t z = 0; z < zSize; ++z ) {
      #endif
         for( cell_idx_t y = 0; y != ySize; ++y )
         {
            using namespace stencil;

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_C  = src->get( x, y, z, Stencil::idx[C]  );
               const real_t dd_tmp_N  = src->get( x, y, z, Stencil::idx[N]  );
               const real_t dd_tmp_S  = src->get( x, y, z, Stencil::idx[S]  );
               const real_t dd_tmp_W  = src->get( x, y, z, Stencil::idx[W]  );
               const real_t dd_tmp_E  = src->get( x, y, z, Stencil::idx[E]  );
               const real_t dd_tmp_T  = src->get( x, y, z, Stencil::idx[T]  );
               const real_t dd_tmp_B  = src->get( x, y, z, Stencil::idx[B]  );
               const real_t dd_tmp_NW = src->get( x, y, z, Stencil::idx[NW] );
               const real_t dd_tmp_NE = src->get( x, y, z, Stencil::idx[NE] );
               const real_t dd_tmp_SW = src->get( x, y, z, Stencil::idx[SW] );
               const real_t dd_tmp_SE = src->get( x, y, z, Stencil::idx[SE] );
               const real_t dd_tmp_TN = src->get( x, y, z, Stencil::idx[TN] );
               const real_t dd_tmp_TS = src->get( x, y, z, Stencil::idx[TS] );
               const real_t dd_tmp_TW = src->get( x, y, z, Stencil::idx[TW] );
               const real_t dd_tmp_TE = src->get( x, y, z, Stencil::idx[TE] );
               const real_t dd_tmp_BN = src->get( x, y, z, Stencil::idx[BN] );
               const real_t dd_tmp_BS = src->get( x, y, z, Stencil::idx[BS] );
               const real_t dd_tmp_BW = src->get( x, y, z, Stencil::idx[BW] );
               const real_t dd_tmp_BE = src->get( x, y, z, Stencil::idx[BE] );

               const real_t velX_trm = dd_tmp_E + dd_tmp_NE + dd_tmp_SE + dd_tmp_TE + dd_tmp_BE;
               const real_t velY_trm = dd_tmp_N + dd_tmp_NW + dd_tmp_TN + dd_tmp_BN;
               const real_t velZ_trm = dd_tmp_T + dd_tmp_TS + dd_tmp_TW;

               const real_t rho = dd_tmp_C + dd_tmp_S + dd_tmp_W + dd_tmp_B + dd_tmp_SW + dd_tmp_BS + dd_tmp_BW + velX_trm + velY_trm + velZ_trm;

               velX[x] = velX_trm - dd_tmp_W  - dd_tmp_NW - dd_tmp_SW - dd_tmp_TW - dd_tmp_BW;
               velY[x] = velY_trm + dd_tmp_NE - dd_tmp_S  - dd_tmp_SW - dd_tmp_SE - dd_tmp_TS - dd_tmp_BS;
               velZ[x] = velZ_trm + dd_tmp_TN + dd_tmp_TE - dd_tmp_B  - dd_tmp_BN - dd_tmp_BS - dd_tmp_BW - dd_tmp_BE;

               feq_common[x] = rho - real_t(1.5) * ( velX[x] * velX[x] + velY[x] * velY[x] + velZ[x] * velZ[x] );

               src->get( x, y, z, Stencil::idx[C] ) = dd_tmp_C * (real_t(1.0) - lambda_e) + lambda_e * t0 * feq_common[x];
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_NE = src->get( x, y, z, Stencil::idx[NE]);
               const real_t dd_tmp_SW = src->get( x, y, z, Stencil::idx[SW]);

               const real_t velXPY = velX[x] + velY[x];
               const real_t  sym_NE_SW = lambda_e_scaled * ( dd_tmp_NE + dd_tmp_SW - fac2 * velXPY * velXPY - t2x2 * feq_common[x] );
               const real_t asym_NE_SW = lambda_d_scaled * ( dd_tmp_NE - dd_tmp_SW - real_t(3.0) * t2x2 * velXPY );

               src->get( x, y, z, Stencil::idx[NE] ) = dd_tmp_NE - sym_NE_SW - asym_NE_SW;
               src->get( x, y, z, Stencil::idx[SW] ) = dd_tmp_SW - sym_NE_SW + asym_NE_SW;
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_SE = src->get( x, y, z, Stencil::idx[SE]);
               const real_t dd_tmp_NW = src->get( x, y, z, Stencil::idx[NW]);

               const real_t velXMY = velX[x] - velY[x];
               const real_t  sym_SE_NW = lambda_e_scaled * ( dd_tmp_SE + dd_tmp_NW - fac2 * velXMY * velXMY - t2x2 * feq_common[x] );
               const real_t asym_SE_NW = lambda_d_scaled * ( dd_tmp_SE - dd_tmp_NW - real_t(3.0) * t2x2 * velXMY );

               src->get( x, y, z, Stencil::idx[SE] ) = dd_tmp_SE - sym_SE_NW - asym_SE_NW;
               src->get( x, y, z, Stencil::idx[NW] ) = dd_tmp_NW - sym_SE_NW + asym_SE_NW;
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_TE = src->get( x, y, z, Stencil::idx[TE]);
               const real_t dd_tmp_BW = src->get( x, y, z, Stencil::idx[BW]);

               const real_t velXPZ = velX[x] + velZ[x];
               const real_t  sym_TE_BW = lambda_e_scaled * ( dd_tmp_TE + dd_tmp_BW - fac2 * velXPZ * velXPZ - t2x2 * feq_common[x] );
               const real_t asym_TE_BW = lambda_d_scaled * ( dd_tmp_TE - dd_tmp_BW - real_t(3.0) * t2x2 * velXPZ );

               src->get( x, y, z, Stencil::idx[TE] ) = dd_tmp_TE - sym_TE_BW - asym_TE_BW;
               src->get( x, y, z, Stencil::idx[BW] ) = dd_tmp_BW - sym_TE_BW + asym_TE_BW;
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_BE = src->get( x, y, z, Stencil::idx[BE]);
               const real_t dd_tmp_TW = src->get( x, y, z, Stencil::idx[TW]);

               const real_t velXMZ = velX[x] - velZ[x];
               const real_t  sym_BE_TW = lambda_e_scaled * ( dd_tmp_BE + dd_tmp_TW - fac2 * velXMZ * velXMZ - t2x2 * feq_common[x] );
               const real_t asym_BE_TW = lambda_d_scaled * ( dd_tmp_BE - dd_tmp_TW - real_t(3.0) * t2x2 * velXMZ );

               src->get( x, y, z, Stencil::idx[BE] ) = dd_tmp_BE - sym_BE_TW - asym_BE_TW;
               src->get( x, y, z, Stencil::idx[TW] ) = dd_tmp_TW - sym_BE_TW + asym_BE_TW;
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_TN = src->get( x, y, z, Stencil::idx[TN]);
               const real_t dd_tmp_BS = src->get( x, y, z, Stencil::idx[BS]);

               const real_t velYPZ = velY[x] + velZ[x];
               const real_t  sym_TN_BS = lambda_e_scaled * ( dd_tmp_TN + dd_tmp_BS - fac2 * velYPZ * velYPZ - t2x2 * feq_common[x] );
               const real_t asym_TN_BS = lambda_d_scaled * ( dd_tmp_TN - dd_tmp_BS - real_t(3.0) * t2x2 * velYPZ );

               src->get( x, y, z, Stencil::idx[TN] ) = dd_tmp_TN - sym_TN_BS - asym_TN_BS;
               src->get( x, y, z, Stencil::idx[BS] ) = dd_tmp_BS - sym_TN_BS + asym_TN_BS;
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_BN = src->get( x, y, z, Stencil::idx[BN]);
               const real_t dd_tmp_TS = src->get( x, y, z, Stencil::idx[TS]);

               const real_t velYMZ = velY[x] - velZ[x];
               const real_t  sym_BN_TS = lambda_e_scaled * ( dd_tmp_BN + dd_tmp_TS - fac2 * velYMZ * velYMZ - t2x2 * feq_common[x] );
               const real_t asym_BN_TS = lambda_d_scaled * ( dd_tmp_BN - dd_tmp_TS - real_t(3.0) * t2x2 * velYMZ );

               src->get( x, y, z, Stencil::idx[BN] ) = dd_tmp_BN - sym_BN_TS - asym_BN_TS;
               src->get( x, y, z, Stencil::idx[TS] ) = dd_tmp_TS - sym_BN_TS + asym_BN_TS;
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_N  = src->get( x, y, z, Stencil::idx[N]);
               const real_t dd_tmp_S  = src->get( x, y, z, Stencil::idx[S]);

               const real_t  sym_N_S = lambda_e_scaled * ( dd_tmp_N + dd_tmp_S - fac1 * velY[x] * velY[x] - t1x2 * feq_common[x] );
               const real_t asym_N_S = lambda_d_scaled * ( dd_tmp_N - dd_tmp_S - real_t(3.0) * t1x2 * velY[x] );

               src->get( x, y, z, Stencil::idx[N] ) = dd_tmp_N - sym_N_S - asym_N_S;
               src->get( x, y, z, Stencil::idx[S] ) = dd_tmp_S - sym_N_S + asym_N_S;
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_E  = src->get( x, y, z, Stencil::idx[E]);
               const real_t dd_tmp_W  = src->get( x, y, z, Stencil::idx[W]);

               const real_t  sym_E_W = lambda_e_scaled * ( dd_tmp_E + dd_tmp_W - fac1 * velX[x] * velX[x] - t1x2 * feq_common[x] );
               const real_t asym_E_W = lambda_d_scaled * ( dd_tmp_E - dd_tmp_W - real_t(3.0) * t1x2 * velX[x] );

               src->get( x, y, z, Stencil::idx[E] ) = dd_tmp_E - sym_E_W - asym_E_W;
               src->get( x, y, z, Stencil::idx[W] ) = dd_tmp_W - sym_E_W + asym_E_W;
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_T  = src->get( x, y, z, Stencil::idx[T]);
               const real_t dd_tmp_B  = src->get( x, y, z, Stencil::idx[B]);

               const real_t  sym_T_B = lambda_e_scaled * ( dd_tmp_T + dd_tmp_B  - fac1 * velZ[x] * velZ[x] - t1x2 * feq_common[x] );
               const real_t asym_T_B = lambda_d_scaled * ( dd_tmp_T - dd_tmp_B - real_t(3.0) * t1x2 * velZ[x] );

               src->get( x, y, z, Stencil::idx[T] ) = dd_tmp_T - sym_T_B - asym_T_B;
               src->get( x, y, z, Stencil::idx[B] ) = dd_tmp_B - sym_T_B + asym_T_B;
            }
         }
      }
   }

   delete[] velX;
   delete[] velY;
   delete[] velZ;
   delete[] feq_common;

#ifdef _OPENMP
   }
#endif
}



///////////////////////////////
// Specialization for D3Q19: //
// - compressible            //
// - no additional forces    //
///////////////////////////////

template< typename LatticeModel_T >
class SplitPureSweep< LatticeModel_T, typename std::enable_if< std::is_same< typename LatticeModel_T::CollisionModel::tag, collision_model::TRT_tag >::value &&
                                                               std::is_same< typename LatticeModel_T::Stencil, stencil::D3Q19 >::value &&
                                                               LatticeModel_T::compressible &&
                                                               std::is_same< typename LatticeModel_T::ForceModel::tag, force_model::None_tag >::value
                                                               >::type > :
   public SweepBase< LatticeModel_T >
{
public:

   static_assert( (std::is_same< typename LatticeModel_T::CollisionModel::tag, collision_model::TRT_tag >::value), "Only works with TRT!" );
   static_assert( (std::is_same< typename LatticeModel_T::Stencil, stencil::D3Q19 >::value),                       "Only works with D3Q19!" );
   static_assert( LatticeModel_T::compressible,                                                                      "Only works with compressible models!" );
   static_assert( (std::is_same< typename LatticeModel_T::ForceModel::tag, force_model::None_tag >::value),        "Only works without additional forces!" );
   static_assert( LatticeModel_T::equilibriumAccuracyOrder == 2, "Only works for lattice models that require the equilibrium distribution to be order 2 accurate!" );

   typedef typename SweepBase<LatticeModel_T>::PdfField_T  PdfField_T;
   typedef typename LatticeModel_T::Stencil                Stencil;

   // block has NO dst pdf field
   SplitPureSweep( const BlockDataID & pdfField ) :
      SweepBase<LatticeModel_T>( pdfField ) {}

   // every block has a dedicated dst pdf field
   SplitPureSweep( const BlockDataID & src, const BlockDataID & dst ) :
      SweepBase<LatticeModel_T>( src, dst ) {}

   void operator()( IBlock * const block );

   void stream ( IBlock * const block, const uint_t numberOfGhostLayersToInclude = uint_t(0) );
   void collide( IBlock * const block, const uint_t numberOfGhostLayersToInclude = uint_t(0) );
};

template< typename LatticeModel_T >
void SplitPureSweep< LatticeModel_T, typename std::enable_if< std::is_same< typename LatticeModel_T::CollisionModel::tag, collision_model::TRT_tag >::value &&
                                                              std::is_same< typename LatticeModel_T::Stencil, stencil::D3Q19 >::value &&
                                                              LatticeModel_T::compressible,
                                                              std::is_same< typename LatticeModel_T::ForceModel::tag, force_model::None_tag >::value
                                                              >::type
   >::operator()( IBlock * const block )
{
   PdfField_T * src( NULL );
   PdfField_T * dst( NULL );

   this->getFields( block, src, dst );

   WALBERLA_ASSERT_NOT_NULLPTR( src );
   WALBERLA_ASSERT_NOT_NULLPTR( dst );

   WALBERLA_ASSERT_GREATER_EQUAL( src->nrOfGhostLayers(), 1 );

   // constants used during stream/collide

   const real_t lambda_e =  src->latticeModel().collisionModel().lambda_e();
   const real_t lambda_d =  src->latticeModel().collisionModel().lambda_d();

   // common prefactors for calculating the equilibrium parts
   const real_t t0_0   = real_t(1.0) / real_t(3.0);                 // 1/3      for C
   const real_t t1x2_0 = real_t(1.0) / real_t(18.0) * real_t(2.0);  // 1/18 * 2 for N, S, W, E, T, B
   const real_t t2x2_0 = real_t(1.0) / real_t(36.0) * real_t(2.0);  // 1/36 * 2 else

   const real_t inv2csq2 = real_t(1.0) / ( real_t(2.0) * ( real_t(1.0) / real_t(3.0) ) * ( real_t(1.0) / real_t(3.0) ) ); //speed of sound related factor for equilibrium distribution function

   // relaxation parameter variables
   const real_t lambda_e_scaled = real_t(0.5) * lambda_e; // 0.5 times the usual value ...
   const real_t lambda_d_scaled = real_t(0.5) * lambda_d; // ... due to the way of calculations

   // loop constants

   const cell_idx_t xSize = cell_idx_c( src->xSize() );
   const cell_idx_t ySize = cell_idx_c( src->ySize() );
   const cell_idx_t zSize = cell_idx_c( src->zSize() );

#ifdef _OPENMP
   #pragma omp parallel
   {
#endif
   // temporaries, calculated by the first innermost loop

   real_t* velX = new real_t[ uint_c(xSize) ];
   real_t* velY = new real_t[ uint_c(xSize) ];
   real_t* velZ = new real_t[ uint_c(xSize) ];

   real_t* t1x2 = new real_t[ uint_c(xSize) ];
   real_t* t2x2 = new real_t[ uint_c(xSize) ];
   real_t* fac1 = new real_t[ uint_c(xSize) ];
   real_t* fac2 = new real_t[ uint_c(xSize) ];

   real_t* feq_common = new real_t[ uint_c(xSize) ];

   if( src->layout() == field::fzyx && dst->layout() == field::fzyx )
   {
      #ifdef _OPENMP
      const int izSize = int_c( zSize );
      #pragma omp for schedule(static)
      for( int iz = 0; iz < izSize; ++iz ) {
         cell_idx_t z = cell_idx_c( iz );
      #else
      for( cell_idx_t z = 0; z < zSize; ++z ) {
      #endif
         for( cell_idx_t y = 0; y != ySize; ++y )
         {
            using namespace stencil;

            real_t* pNE = &src->get(-1, y-1, z  , Stencil::idx[NE]);
            real_t* pN  = &src->get(0 , y-1, z  , Stencil::idx[N]);
            real_t* pNW = &src->get(+1, y-1, z  , Stencil::idx[NW]);
            real_t* pW  = &src->get(+1, y  , z  , Stencil::idx[W]);
            real_t* pSW = &src->get(+1, y+1, z  , Stencil::idx[SW]);
            real_t* pS  = &src->get(0 , y+1, z  , Stencil::idx[S]);
            real_t* pSE = &src->get(-1, y+1, z  , Stencil::idx[SE]);
            real_t* pE  = &src->get(-1, y  , z  , Stencil::idx[E]);
            real_t* pT  = &src->get(0 , y  , z-1, Stencil::idx[T]);
            real_t* pTE = &src->get(-1, y  , z-1, Stencil::idx[TE]);
            real_t* pTN = &src->get(0 , y-1, z-1, Stencil::idx[TN]);
            real_t* pTW = &src->get(+1, y  , z-1, Stencil::idx[TW]);
            real_t* pTS = &src->get(0 , y+1, z-1, Stencil::idx[TS]);
            real_t* pB  = &src->get(0 , y  , z+1, Stencil::idx[B]);
            real_t* pBE = &src->get(-1, y  , z+1, Stencil::idx[BE]);
            real_t* pBN = &src->get(0 , y-1, z+1, Stencil::idx[BN]);
            real_t* pBW = &src->get(+1, y  , z+1, Stencil::idx[BW]);
            real_t* pBS = &src->get(0 , y+1, z+1, Stencil::idx[BS]);
            real_t* pC  = &src->get(0 , y  , z  , Stencil::idx[C]);

            real_t* dC = &dst->get(0,y,z,Stencil::idx[C]);

            __alignx( 32, velX );
            __alignx( 32, velY );
            __alignx( 32, velZ );
            __alignx( 32, t1x2 );
            __alignx( 32, t2x2 );
            __alignx( 32, fac1 );
            __alignx( 32, fac2 );
            __alignx( 32, feq_common  );

            __alignx( 32, pNE );
            __alignx( 32, pN  );
            __alignx( 32, pNW );
            __alignx( 32, pW  );
            __alignx( 32, pSW );
            __alignx( 32, pS  );
            __alignx( 32, pSE );
            __alignx( 32, pE  );
            __alignx( 32, pT  );
            __alignx( 32, pTE );
            __alignx( 32, pTN );
            __alignx( 32, pTW );
            __alignx( 32, pTS );
            __alignx( 32, pB  );
            __alignx( 32, pBE );
            __alignx( 32, pBN );
            __alignx( 32, pBW );
            __alignx( 32, pBS );
            __alignx( 32, pC  );

            __alignx( 32, dC  );

            #pragma disjoint( *velX, *velY, *velZ, *t1x2, *t2x2, *fac1, *fac2, *feq_common, *pNE, *pN, *pNW, *pW, *pSW, *pS, *pSE, \
                              *pE, *pT, *pTE, *pTN, *pTW, *pTS, *pB, *pBE, *pBN, *pBW, *pBS, *pC, *dC, \
                              t0_0, t1x2_0, t2x2_0, inv2csq2, lambda_e_scaled, lambda_d_scaled )

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t velX_trm = pE[x] + pNE[x] + pSE[x] + pTE[x] + pBE[x];
               const real_t velY_trm = pN[x] + pNW[x] + pTN[x] + pBN[x];
               const real_t velZ_trm = pT[x] + pTS[x] + pTW[x];

               const real_t rho = pC[x] + pS[x] + pW[x] + pB[x] + pSW[x] + pBS[x] + pBW[x] + velX_trm + velY_trm + velZ_trm;
               const real_t invRho = real_t(1.0) / rho;

               velX[x] = invRho * ( velX_trm - pW[x]  - pNW[x] - pSW[x] - pTW[x] - pBW[x] );
               velY[x] = invRho * ( velY_trm + pNE[x] - pS[x]  - pSW[x] - pSE[x] - pTS[x] - pBS[x] );
               velZ[x] = invRho * ( velZ_trm + pTN[x] + pTE[x] - pB[x]  - pBN[x] - pBS[x] - pBW[x] - pBE[x] );

               t1x2[x] = t1x2_0 * rho;
               t2x2[x] = t2x2_0 * rho;
               fac1[x] = t1x2_0 * rho * inv2csq2;
               fac2[x] = t2x2_0 * rho * inv2csq2;

               feq_common[x] = real_t(1.0) - real_t(1.5) * ( velX[x] * velX[x] + velY[x] * velY[x] + velZ[x] * velZ[x] );

               dC[x] = pC[x] * (real_t(1.0) - lambda_e) + lambda_e * t0_0 * rho * feq_common[x];
            }

            real_t* dNE = &dst->get(0,y,z,Stencil::idx[NE]);
            real_t* dSW = &dst->get(0,y,z,Stencil::idx[SW]);

            __alignx( 32, dNE );
            __alignx( 32, dSW );

            #pragma disjoint( *dNE, *dSW, \
                              *velX, *velY, *velZ, *t1x2, *t2x2, *fac1, *fac2, *feq_common, *pNE, *pN, *pNW, *pW, *pSW, *pS, *pSE, \
                              *pE, *pT, *pTE, *pTN, *pTW, *pTS, *pB, *pBE, *pBN, *pBW, *pBS, *pC, *dC, \
                              t0_0, t1x2_0, t2x2_0, inv2csq2, lambda_e_scaled, lambda_d_scaled )

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t velXPY = velX[x] + velY[x];
               const real_t  sym_NE_SW = lambda_e_scaled * ( pNE[x] + pSW[x] - fac2[x] * velXPY * velXPY - t2x2[x] * feq_common[x] );
               const real_t asym_NE_SW = lambda_d_scaled * ( pNE[x] - pSW[x] - real_t(3.0) * t2x2[x] * velXPY );

               dNE[x] = pNE[x] - sym_NE_SW - asym_NE_SW;
               dSW[x] = pSW[x] - sym_NE_SW + asym_NE_SW;
            }

            real_t* dSE = &dst->get(0,y,z,Stencil::idx[SE]);
            real_t* dNW = &dst->get(0,y,z,Stencil::idx[NW]);

            __alignx( 32, dSE );
            __alignx( 32, dNW );

            #pragma disjoint( *dSE, *dNW, *dNE, *dSW, \
                              *velX, *velY, *velZ, *t1x2, *t2x2, *fac1, *fac2, *feq_common, *pNE, *pN, *pNW, *pW, *pSW, *pS, *pSE, \
                              *pE, *pT, *pTE, *pTN, *pTW, *pTS, *pB, *pBE, *pBN, *pBW, *pBS, *pC, *dC, \
                              t0_0, t1x2_0, t2x2_0, inv2csq2, lambda_e_scaled, lambda_d_scaled )

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t velXMY = velX[x] - velY[x];
               const real_t  sym_SE_NW = lambda_e_scaled * ( pSE[x] + pNW[x] - fac2[x] * velXMY * velXMY - t2x2[x] * feq_common[x] );
               const real_t asym_SE_NW = lambda_d_scaled * ( pSE[x] - pNW[x] - real_t(3.0) * t2x2[x] * velXMY );

               dSE[x] = pSE[x] - sym_SE_NW - asym_SE_NW;
               dNW[x] = pNW[x] - sym_SE_NW + asym_SE_NW;
            }

            real_t* dTE = &dst->get(0,y,z,Stencil::idx[TE]);
            real_t* dBW = &dst->get(0,y,z,Stencil::idx[BW]);

            __alignx( 32, dTE );
            __alignx( 32, dBW );

            #pragma disjoint( *dTE, *dBW, *dSE, *dNW, *dNE, *dSW, \
                              *velX, *velY, *velZ, *t1x2, *t2x2, *fac1, *fac2, *feq_common, *pNE, *pN, *pNW, *pW, *pSW, *pS, *pSE, \
                              *pE, *pT, *pTE, *pTN, *pTW, *pTS, *pB, *pBE, *pBN, *pBW, *pBS, *pC, *dC, \
                              t0_0, t1x2_0, t2x2_0, inv2csq2, lambda_e_scaled, lambda_d_scaled )

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t velXPZ = velX[x] + velZ[x];
               const real_t  sym_TE_BW = lambda_e_scaled * ( pTE[x] + pBW[x] - fac2[x] * velXPZ * velXPZ - t2x2[x] * feq_common[x] );
               const real_t asym_TE_BW = lambda_d_scaled * ( pTE[x] - pBW[x] - real_t(3.0) * t2x2[x] * velXPZ );

               dTE[x] = pTE[x] - sym_TE_BW - asym_TE_BW;
               dBW[x] = pBW[x] - sym_TE_BW + asym_TE_BW;
            }

            real_t* dBE = &dst->get(0,y,z,Stencil::idx[BE]);
            real_t* dTW = &dst->get(0,y,z,Stencil::idx[TW]);

            __alignx( 32, dBE );
            __alignx( 32, dTW );

            #pragma disjoint( *dBE, *dTW, *dTE, *dBW, *dSE, *dNW, *dNE, *dSW, \
                              *velX, *velY, *velZ, *t1x2, *t2x2, *fac1, *fac2, *feq_common, *pNE, *pN, *pNW, *pW, *pSW, *pS, *pSE, \
                              *pE, *pT, *pTE, *pTN, *pTW, *pTS, *pB, *pBE, *pBN, *pBW, *pBS, *pC, *dC, \
                              t0_0, t1x2_0, t2x2_0, inv2csq2, lambda_e_scaled, lambda_d_scaled )

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t velXMZ = velX[x] - velZ[x];
               const real_t  sym_BE_TW = lambda_e_scaled * ( pBE[x] + pTW[x] - fac2[x] * velXMZ * velXMZ - t2x2[x] * feq_common[x] );
               const real_t asym_BE_TW = lambda_d_scaled * ( pBE[x] - pTW[x] - real_t(3.0) * t2x2[x] * velXMZ );

               dBE[x] = pBE[x] - sym_BE_TW - asym_BE_TW;
               dTW[x] = pTW[x] - sym_BE_TW + asym_BE_TW;
            }

            real_t* dTN = &dst->get(0,y,z,Stencil::idx[TN]);
            real_t* dBS = &dst->get(0,y,z,Stencil::idx[BS]);

            __alignx( 32, dTN );
            __alignx( 32, dBS );

            #pragma disjoint( *dTN, *dBS, *dBE, *dTW, *dTE, *dBW, *dSE, *dNW, *dNE, *dSW, \
                              *velX, *velY, *velZ, *t1x2, *t2x2, *fac1, *fac2, *feq_common, *pNE, *pN, *pNW, *pW, *pSW, *pS, *pSE, \
                              *pE, *pT, *pTE, *pTN, *pTW, *pTS, *pB, *pBE, *pBN, *pBW, *pBS, *pC, *dC, \
                              t0_0, t1x2_0, t2x2_0, inv2csq2, lambda_e_scaled, lambda_d_scaled )

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t velYPZ = velY[x] + velZ[x];
               const real_t  sym_TN_BS = lambda_e_scaled * ( pTN[x] + pBS[x] - fac2[x] * velYPZ * velYPZ - t2x2[x] * feq_common[x] );
               const real_t asym_TN_BS = lambda_d_scaled * ( pTN[x] - pBS[x] - real_t(3.0) * t2x2[x] * velYPZ );

               dTN[x] = pTN[x] - sym_TN_BS - asym_TN_BS;
               dBS[x] = pBS[x] - sym_TN_BS + asym_TN_BS;
            }

            real_t* dBN = &dst->get(0,y,z,Stencil::idx[BN]);
            real_t* dTS = &dst->get(0,y,z,Stencil::idx[TS]);

            __alignx( 32, dBN );
            __alignx( 32, dTS );

            #pragma disjoint( *dBN, *dTS, *dTN, *dBS, *dBE, *dTW, *dTE, *dBW, *dSE, *dNW, *dNE, *dSW, \
                              *velX, *velY, *velZ, *t1x2, *t2x2, *fac1, *fac2, *feq_common, *pNE, *pN, *pNW, *pW, *pSW, *pS, *pSE, \
                              *pE, *pT, *pTE, *pTN, *pTW, *pTS, *pB, *pBE, *pBN, *pBW, *pBS, *pC, *dC, \
                              t0_0, t1x2_0, t2x2_0, inv2csq2, lambda_e_scaled, lambda_d_scaled )

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t velYMZ = velY[x] - velZ[x];
               const real_t  sym_BN_TS = lambda_e_scaled * ( pBN[x] + pTS[x] - fac2[x] * velYMZ * velYMZ - t2x2[x] * feq_common[x] );
               const real_t asym_BN_TS = lambda_d_scaled * ( pBN[x] - pTS[x] - real_t(3.0) * t2x2[x] * velYMZ );

               dBN[x] = pBN[x] - sym_BN_TS - asym_BN_TS;
               dTS[x] = pTS[x] - sym_BN_TS + asym_BN_TS;
            }

            real_t* dN = &dst->get(0,y,z,Stencil::idx[N]);
            real_t* dS = &dst->get(0,y,z,Stencil::idx[S]);

            __alignx( 32, dN );
            __alignx( 32, dS );

            #pragma disjoint( *dN, *dS, *dBN, *dTS, *dTN, *dBS, *dBE, *dTW, *dTE, *dBW, *dSE, *dNW, *dNE, *dSW, \
                              *velX, *velY, *velZ, *t1x2, *t2x2, *fac1, *fac2, *feq_common, *pNE, *pN, *pNW, *pW, *pSW, *pS, *pSE, \
                              *pE, *pT, *pTE, *pTN, *pTW, *pTS, *pB, *pBE, *pBN, *pBW, *pBS, *pC, *dC, \
                              t0_0, t1x2_0, t2x2_0, inv2csq2, lambda_e_scaled, lambda_d_scaled )

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t  sym_N_S = lambda_e_scaled * ( pN[x] + pS[x] - fac1[x] * velY[x] * velY[x] - t1x2[x] * feq_common[x] );
               const real_t asym_N_S = lambda_d_scaled * ( pN[x] - pS[x] - real_t(3.0) * t1x2[x] * velY[x] );

               dN[x] = pN[x] - sym_N_S - asym_N_S;
               dS[x] = pS[x] - sym_N_S + asym_N_S;
            }

            real_t* dE = &dst->get(0,y,z,Stencil::idx[E]);
            real_t* dW = &dst->get(0,y,z,Stencil::idx[W]);

            __alignx( 32, dE );
            __alignx( 32, dW );

            #pragma disjoint( *dE, *dW, *dN, *dS, *dBN, *dTS, *dTN, *dBS, *dBE, *dTW, *dTE, *dBW, *dSE, *dNW, *dNE, *dSW, \
                              *velX, *velY, *velZ, *t1x2, *t2x2, *fac1, *fac2, *feq_common, *pNE, *pN, *pNW, *pW, *pSW, *pS, *pSE, \
                              *pE, *pT, *pTE, *pTN, *pTW, *pTS, *pB, *pBE, *pBN, *pBW, *pBS, *pC, *dC, \
                              t0_0, t1x2_0, t2x2_0, inv2csq2, lambda_e_scaled, lambda_d_scaled )

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t  sym_E_W = lambda_e_scaled * ( pE[x] + pW[x] - fac1[x] * velX[x] * velX[x] - t1x2[x] * feq_common[x] );
               const real_t asym_E_W = lambda_d_scaled * ( pE[x] - pW[x] - real_t(3.0) * t1x2[x] * velX[x] );

               dE[x] = pE[x] - sym_E_W - asym_E_W;
               dW[x] = pW[x] - sym_E_W + asym_E_W;
            }

            real_t* dT = &dst->get(0,y,z,Stencil::idx[T]);
            real_t* dB = &dst->get(0,y,z,Stencil::idx[B]);

            __alignx( 32, dT );
            __alignx( 32, dB );

            #pragma disjoint( *dT, *dB, *dE, *dW, *dN, *dS, *dBN, *dTS, *dTN, *dBS, *dBE, *dTW, *dTE, *dBW, *dSE, *dNW, *dNE, *dSW, \
                              *velX, *velY, *velZ, *t1x2, *t2x2, *fac1, *fac2, *feq_common, *pNE, *pN, *pNW, *pW, *pSW, *pS, *pSE, \
                              *pE, *pT, *pTE, *pTN, *pTW, *pTS, *pB, *pBE, *pBN, *pBW, *pBS, *pC, *dC, \
                              t0_0, t1x2_0, t2x2_0, inv2csq2, lambda_e_scaled, lambda_d_scaled )

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t  sym_T_B = lambda_e_scaled * ( pT[x] + pB[x]  - fac1[x] * velZ[x] * velZ[x] - t1x2[x] * feq_common[x] );
               const real_t asym_T_B = lambda_d_scaled * ( pT[x] - pB[x] - real_t(3.0) * t1x2[x] * velZ[x] );

               dT[x] = pT[x] - sym_T_B - asym_T_B;
               dB[x] = pB[x] - sym_T_B + asym_T_B;
            }
         }
      }
   }
   else // ==> src->layout() == field::zyxf || dst->layout() == field::zyxf
   {
      #ifdef _OPENMP
      const int izSize = int_c( zSize );
      #pragma omp for schedule(static)
      for( int iz = 0; iz < izSize; ++iz ) {
         cell_idx_t z = cell_idx_c( iz );
      #else
      for( cell_idx_t z = 0; z < zSize; ++z ) {
      #endif
         for( cell_idx_t y = 0; y != ySize; ++y )
         {
            using namespace stencil;

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_NE = src->get(x-1, y-1, z  , Stencil::idx[NE]);
               const real_t dd_tmp_N  = src->get(x  , y-1, z  , Stencil::idx[N]);
               const real_t dd_tmp_NW = src->get(x+1, y-1, z  , Stencil::idx[NW]);
               const real_t dd_tmp_W  = src->get(x+1, y  , z  , Stencil::idx[W]);
               const real_t dd_tmp_SW = src->get(x+1, y+1, z  , Stencil::idx[SW]);
               const real_t dd_tmp_S  = src->get(x  , y+1, z  , Stencil::idx[S]);
               const real_t dd_tmp_SE = src->get(x-1, y+1, z  , Stencil::idx[SE]);
               const real_t dd_tmp_E  = src->get(x-1, y  , z  , Stencil::idx[E]);
               const real_t dd_tmp_T  = src->get(x  , y  , z-1, Stencil::idx[T]);
               const real_t dd_tmp_TE = src->get(x-1, y  , z-1, Stencil::idx[TE]);
               const real_t dd_tmp_TN = src->get(x  , y-1, z-1, Stencil::idx[TN]);
               const real_t dd_tmp_TW = src->get(x+1, y  , z-1, Stencil::idx[TW]);
               const real_t dd_tmp_TS = src->get(x  , y+1, z-1, Stencil::idx[TS]);
               const real_t dd_tmp_B  = src->get(x  , y  , z+1, Stencil::idx[B]);
               const real_t dd_tmp_BE = src->get(x-1, y  , z+1, Stencil::idx[BE]);
               const real_t dd_tmp_BN = src->get(x  , y-1, z+1, Stencil::idx[BN]);
               const real_t dd_tmp_BW = src->get(x+1, y  , z+1, Stencil::idx[BW]);
               const real_t dd_tmp_BS = src->get(x  , y+1, z+1, Stencil::idx[BS]);
               const real_t dd_tmp_C  = src->get(x  , y  , z  , Stencil::idx[C]);

               const real_t velX_trm = dd_tmp_E + dd_tmp_NE + dd_tmp_SE + dd_tmp_TE + dd_tmp_BE;
               const real_t velY_trm = dd_tmp_N + dd_tmp_NW + dd_tmp_TN + dd_tmp_BN;
               const real_t velZ_trm = dd_tmp_T + dd_tmp_TS + dd_tmp_TW;

               const real_t rho = dd_tmp_C + dd_tmp_S + dd_tmp_W + dd_tmp_B + dd_tmp_SW + dd_tmp_BS + dd_tmp_BW + velX_trm + velY_trm + velZ_trm;
               const real_t invRho = real_t(1.0) / rho;

               velX[x] = invRho * ( velX_trm - dd_tmp_W  - dd_tmp_NW - dd_tmp_SW - dd_tmp_TW - dd_tmp_BW );
               velY[x] = invRho * ( velY_trm + dd_tmp_NE - dd_tmp_S  - dd_tmp_SW - dd_tmp_SE - dd_tmp_TS - dd_tmp_BS );
               velZ[x] = invRho * ( velZ_trm + dd_tmp_TN + dd_tmp_TE - dd_tmp_B  - dd_tmp_BN - dd_tmp_BS - dd_tmp_BW - dd_tmp_BE );

               t1x2[x] = t1x2_0 * rho;
               t2x2[x] = t2x2_0 * rho;
               fac1[x] = t1x2_0 * rho * inv2csq2;
               fac2[x] = t2x2_0 * rho * inv2csq2;

               feq_common[x] = real_t(1.0) - real_t(1.5) * ( velX[x] * velX[x] + velY[x] * velY[x] + velZ[x] * velZ[x] );

               dst->get( x, y, z, Stencil::idx[C] ) = dd_tmp_C * (real_t(1.0) - lambda_e) + lambda_e * t0_0 * rho * feq_common[x];
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_NE = src->get(x-1, y-1, z, Stencil::idx[NE]);
               const real_t dd_tmp_SW = src->get(x+1, y+1, z, Stencil::idx[SW]);

               const real_t velXPY = velX[x] + velY[x];
               const real_t  sym_NE_SW = lambda_e_scaled * ( dd_tmp_NE + dd_tmp_SW - fac2[x] * velXPY * velXPY - t2x2[x] * feq_common[x] );
               const real_t asym_NE_SW = lambda_d_scaled * ( dd_tmp_NE - dd_tmp_SW - real_t(3.0) * t2x2[x] * velXPY );

               dst->get( x, y, z, Stencil::idx[NE] ) = dd_tmp_NE - sym_NE_SW - asym_NE_SW;
               dst->get( x, y, z, Stencil::idx[SW] ) = dd_tmp_SW - sym_NE_SW + asym_NE_SW;
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_SE = src->get(x-1, y+1, z, Stencil::idx[SE]);
               const real_t dd_tmp_NW = src->get(x+1, y-1, z, Stencil::idx[NW]);

               const real_t velXMY = velX[x] - velY[x];
               const real_t  sym_SE_NW = lambda_e_scaled * ( dd_tmp_SE + dd_tmp_NW - fac2[x] * velXMY * velXMY - t2x2[x] * feq_common[x] );
               const real_t asym_SE_NW = lambda_d_scaled * ( dd_tmp_SE - dd_tmp_NW - real_t(3.0) * t2x2[x] * velXMY );

               dst->get( x, y, z, Stencil::idx[SE] ) = dd_tmp_SE - sym_SE_NW - asym_SE_NW;
               dst->get( x, y, z, Stencil::idx[NW] ) = dd_tmp_NW - sym_SE_NW + asym_SE_NW;
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_TE = src->get(x-1, y, z-1, Stencil::idx[TE]);
               const real_t dd_tmp_BW = src->get(x+1, y, z+1, Stencil::idx[BW]);

               const real_t velXPZ = velX[x] + velZ[x];
               const real_t  sym_TE_BW = lambda_e_scaled * ( dd_tmp_TE + dd_tmp_BW - fac2[x] * velXPZ * velXPZ - t2x2[x] * feq_common[x] );
               const real_t asym_TE_BW = lambda_d_scaled * ( dd_tmp_TE - dd_tmp_BW - real_t(3.0) * t2x2[x] * velXPZ );

               dst->get( x, y, z, Stencil::idx[TE] ) = dd_tmp_TE - sym_TE_BW - asym_TE_BW;
               dst->get( x, y, z, Stencil::idx[BW] ) = dd_tmp_BW - sym_TE_BW + asym_TE_BW;
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_BE = src->get(x-1, y, z+1, Stencil::idx[BE]);
               const real_t dd_tmp_TW = src->get(x+1, y, z-1, Stencil::idx[TW]);

               const real_t velXMZ = velX[x] - velZ[x];
               const real_t  sym_BE_TW = lambda_e_scaled * ( dd_tmp_BE + dd_tmp_TW - fac2[x] * velXMZ * velXMZ - t2x2[x] * feq_common[x] );
               const real_t asym_BE_TW = lambda_d_scaled * ( dd_tmp_BE - dd_tmp_TW - real_t(3.0) * t2x2[x] * velXMZ );

               dst->get( x, y, z, Stencil::idx[BE] ) = dd_tmp_BE - sym_BE_TW - asym_BE_TW;
               dst->get( x, y, z, Stencil::idx[TW] ) = dd_tmp_TW - sym_BE_TW + asym_BE_TW;
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_TN = src->get(x, y-1, z-1, Stencil::idx[TN]);
               const real_t dd_tmp_BS = src->get(x, y+1, z+1, Stencil::idx[BS]);

               const real_t velYPZ = velY[x] + velZ[x];
               const real_t  sym_TN_BS = lambda_e_scaled * ( dd_tmp_TN + dd_tmp_BS - fac2[x] * velYPZ * velYPZ - t2x2[x] * feq_common[x] );
               const real_t asym_TN_BS = lambda_d_scaled * ( dd_tmp_TN - dd_tmp_BS - real_t(3.0) * t2x2[x] * velYPZ );

               dst->get( x, y, z, Stencil::idx[TN] ) = dd_tmp_TN - sym_TN_BS - asym_TN_BS;
               dst->get( x, y, z, Stencil::idx[BS] ) = dd_tmp_BS - sym_TN_BS + asym_TN_BS;
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_BN = src->get(x, y-1, z+1, Stencil::idx[BN]);
               const real_t dd_tmp_TS = src->get(x, y+1, z-1, Stencil::idx[TS]);

               const real_t velYMZ = velY[x] - velZ[x];
               const real_t  sym_BN_TS = lambda_e_scaled * ( dd_tmp_BN + dd_tmp_TS - fac2[x] * velYMZ * velYMZ - t2x2[x] * feq_common[x] );
               const real_t asym_BN_TS = lambda_d_scaled * ( dd_tmp_BN - dd_tmp_TS - real_t(3.0) * t2x2[x] * velYMZ );

               dst->get( x, y, z, Stencil::idx[BN] ) = dd_tmp_BN - sym_BN_TS - asym_BN_TS;
               dst->get( x, y, z, Stencil::idx[TS] ) = dd_tmp_TS - sym_BN_TS + asym_BN_TS;
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_N  = src->get(x, y-1, z, Stencil::idx[N]);
               const real_t dd_tmp_S  = src->get(x, y+1, z, Stencil::idx[S]);

               const real_t  sym_N_S = lambda_e_scaled * ( dd_tmp_N + dd_tmp_S - fac1[x] * velY[x] * velY[x] - t1x2[x] * feq_common[x] );
               const real_t asym_N_S = lambda_d_scaled * ( dd_tmp_N - dd_tmp_S - real_t(3.0) * t1x2[x] * velY[x] );

               dst->get( x, y, z, Stencil::idx[N] ) = dd_tmp_N - sym_N_S - asym_N_S;
               dst->get( x, y, z, Stencil::idx[S] ) = dd_tmp_S - sym_N_S + asym_N_S;
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_E  = src->get(x-1, y, z, Stencil::idx[E]);
               const real_t dd_tmp_W  = src->get(x+1, y, z, Stencil::idx[W]);

               const real_t  sym_E_W = lambda_e_scaled * ( dd_tmp_E + dd_tmp_W - fac1[x] * velX[x] * velX[x] - t1x2[x] * feq_common[x] );
               const real_t asym_E_W = lambda_d_scaled * ( dd_tmp_E - dd_tmp_W - real_t(3.0) * t1x2[x] * velX[x] );

               dst->get( x, y, z, Stencil::idx[E] ) = dd_tmp_E - sym_E_W - asym_E_W;
               dst->get( x, y, z, Stencil::idx[W] ) = dd_tmp_W - sym_E_W + asym_E_W;
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_T  = src->get(x, y, z-1, Stencil::idx[T]);
               const real_t dd_tmp_B  = src->get(x, y, z+1, Stencil::idx[B]);

               const real_t  sym_T_B = lambda_e_scaled * ( dd_tmp_T + dd_tmp_B  - fac1[x] * velZ[x] * velZ[x] - t1x2[x] * feq_common[x] );
               const real_t asym_T_B = lambda_d_scaled * ( dd_tmp_T - dd_tmp_B - real_t(3.0) * t1x2[x] * velZ[x] );

               dst->get( x, y, z, Stencil::idx[T] ) = dd_tmp_T - sym_T_B - asym_T_B;
               dst->get( x, y, z, Stencil::idx[B] ) = dd_tmp_B - sym_T_B + asym_T_B;
            }
         }
      }
   }

   delete[] velX;
   delete[] velY;
   delete[] velZ;
   delete[] t1x2;
   delete[] t2x2;
   delete[] fac1;
   delete[] fac2;
   delete[] feq_common;

#ifdef _OPENMP
   }
#endif

   src->swapDataPointers( dst );
}

template< typename LatticeModel_T >
void SplitPureSweep< LatticeModel_T, typename std::enable_if< std::is_same< typename LatticeModel_T::CollisionModel::tag, collision_model::TRT_tag >::value &&
                                                              std::is_same< typename LatticeModel_T::Stencil, stencil::D3Q19 >::value &&
                                                              LatticeModel_T::compressible &&
                                                              std::is_same< typename LatticeModel_T::ForceModel::tag &&
                                                              force_model::None_tag
                                                              >::type
   >::stream( IBlock * const block, const uint_t numberOfGhostLayersToInclude )
{
   PdfField_T * src( NULL );
   PdfField_T * dst( NULL );

   this->getFields( block, src, dst );

   StreamEverything< LatticeModel_T >::execute( src, dst, numberOfGhostLayersToInclude );
}

template< typename LatticeModel_T >
void SplitPureSweep< LatticeModel_T, typename std::enable_if< std::is_same< typename LatticeModel_T::CollisionModel::tag, collision_model::TRT_tag >::value &&
                                                                           std::is_same< typename LatticeModel_T::Stencil, stencil::D3Q19 >::value &&
                                                                           LatticeModel_T::compressible &&
                                                                           std::is_same< typename LatticeModel_T::ForceModel::tag, force_model::None_tag >::value
                                                                           >::type
#ifdef NDEBUG
   >::collide( IBlock * const block, const uint_t /*numberOfGhostLayersToInclude*/ )
#else
   >::collide( IBlock * const block, const uint_t numberOfGhostLayersToInclude )
#endif
{
   WALBERLA_ASSERT_EQUAL( numberOfGhostLayersToInclude, uint_t(0) ); // the implementation right now doesn't support inclusion of ghost layers in collide step!

   PdfField_T * src = this->getSrcField( block );

   WALBERLA_ASSERT_NOT_NULLPTR( src );

   WALBERLA_ASSERT_GREATER_EQUAL( src->nrOfGhostLayers(), numberOfGhostLayersToInclude );

   // constants used during stream/collide

   const real_t lambda_e =  src->latticeModel().collisionModel().lambda_e();
   const real_t lambda_d =  src->latticeModel().collisionModel().lambda_d();

   // common prefactors for calculating the equilibrium parts
   const real_t t0_0   = real_t(1.0) / real_t(3.0);                 // 1/3      for C
   const real_t t1x2_0 = real_t(1.0) / real_t(18.0) * real_t(2.0);  // 1/18 * 2 for N, S, W, E, T, B
   const real_t t2x2_0 = real_t(1.0) / real_t(36.0) * real_t(2.0);  // 1/36 * 2 else

   const real_t inv2csq2 = real_t(1.0) / ( real_t(2.0) * ( real_t(1.0) / real_t(3.0) ) * ( real_t(1.0) / real_t(3.0) ) ); //speed of sound related factor for equilibrium distribution function

   // relaxation parameter variables
   const real_t lambda_e_scaled = real_t(0.5) * lambda_e; // 0.5 times the usual value ...
   const real_t lambda_d_scaled = real_t(0.5) * lambda_d; // ... due to the way of calculations

   // loop constants

   const cell_idx_t xSize = cell_idx_c( src->xSize() );
   const cell_idx_t ySize = cell_idx_c( src->ySize() );
   const cell_idx_t zSize = cell_idx_c( src->zSize() );

#ifdef _OPENMP
   #pragma omp parallel
   {
#endif
   // temporaries, calculated by the first innermost loop

   real_t* velX = new real_t[ uint_c(xSize) ];
   real_t* velY = new real_t[ uint_c(xSize) ];
   real_t* velZ = new real_t[ uint_c(xSize) ];

   real_t* t1x2 = new real_t[ uint_c(xSize) ];
   real_t* t2x2 = new real_t[ uint_c(xSize) ];
   real_t* fac1 = new real_t[ uint_c(xSize) ];
   real_t* fac2 = new real_t[ uint_c(xSize) ];

   real_t* feq_common = new real_t[ uint_c(xSize) ];

   if( src->layout() == field::fzyx )
   {
      #ifdef _OPENMP
      const int izSize = int_c( zSize );
      #pragma omp for schedule(static)
      for( int iz = 0; iz < izSize; ++iz ) {
         cell_idx_t z = cell_idx_c( iz );
      #else
      for( cell_idx_t z = 0; z < zSize; ++z ) {
      #endif
         for( cell_idx_t y = 0; y != ySize; ++y )
         {
            using namespace stencil;

            real_t* pC  = &src->get( 0, y, z, Stencil::idx[C]);
            real_t* pN  = &src->get( 0, y, z, Stencil::idx[N]);
            real_t* pS  = &src->get( 0, y, z, Stencil::idx[S]);
            real_t* pW  = &src->get( 0, y, z, Stencil::idx[W]);
            real_t* pE  = &src->get( 0, y, z, Stencil::idx[E]);
            real_t* pT  = &src->get( 0, y, z, Stencil::idx[T]);
            real_t* pB  = &src->get( 0, y, z, Stencil::idx[B]);
            real_t* pNW = &src->get( 0, y, z, Stencil::idx[NW]);
            real_t* pNE = &src->get( 0, y, z, Stencil::idx[NE]);
            real_t* pSW = &src->get( 0, y, z, Stencil::idx[SW]);
            real_t* pSE = &src->get( 0, y, z, Stencil::idx[SE]);
            real_t* pTN = &src->get( 0, y, z, Stencil::idx[TN]);
            real_t* pTS = &src->get( 0, y, z, Stencil::idx[TS]);
            real_t* pTW = &src->get( 0, y, z, Stencil::idx[TW]);
            real_t* pTE = &src->get( 0, y, z, Stencil::idx[TE]);
            real_t* pBN = &src->get( 0, y, z, Stencil::idx[BN]);
            real_t* pBS = &src->get( 0, y, z, Stencil::idx[BS]);
            real_t* pBW = &src->get( 0, y, z, Stencil::idx[BW]);
            real_t* pBE = &src->get( 0, y, z, Stencil::idx[BE]);

            __alignx( 32, velX );
            __alignx( 32, velY );
            __alignx( 32, velZ );
            __alignx( 32, t1x2 );
            __alignx( 32, t2x2 );
            __alignx( 32, fac1 );
            __alignx( 32, fac2 );
            __alignx( 32, feq_common  );

            __alignx( 32, pNE );
            __alignx( 32, pN  );
            __alignx( 32, pNW );
            __alignx( 32, pW  );
            __alignx( 32, pSW );
            __alignx( 32, pS  );
            __alignx( 32, pSE );
            __alignx( 32, pE  );
            __alignx( 32, pT  );
            __alignx( 32, pTE );
            __alignx( 32, pTN );
            __alignx( 32, pTW );
            __alignx( 32, pTS );
            __alignx( 32, pB  );
            __alignx( 32, pBE );
            __alignx( 32, pBN );
            __alignx( 32, pBW );
            __alignx( 32, pBS );
            __alignx( 32, pC  );

            #pragma disjoint( *velX, *velY, *velZ, *t1x2, *t2x2, *fac1, *fac2, *feq_common, *pNE, *pN, *pNW, *pW, *pSW, *pS, *pSE, \
                              *pE, *pT, *pTE, *pTN, *pTW, *pTS, *pB, *pBE, *pBN, *pBW, *pBS, *pC, \
                              t0_0, t1x2_0, t2x2_0, inv2csq2, lambda_e_scaled, lambda_d_scaled )

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t velX_trm = pE[x] + pNE[x] + pSE[x] + pTE[x] + pBE[x];
               const real_t velY_trm = pN[x] + pNW[x] + pTN[x] + pBN[x];
               const real_t velZ_trm = pT[x] + pTS[x] + pTW[x];

               const real_t rho = pC[x] + pS[x] + pW[x] + pB[x] + pSW[x] + pBS[x] + pBW[x] + velX_trm + velY_trm + velZ_trm;
               const real_t invRho = real_t(1.0) / rho;

               velX[x] = invRho * ( velX_trm - pW[x]  - pNW[x] - pSW[x] - pTW[x] - pBW[x] );
               velY[x] = invRho * ( velY_trm + pNE[x] - pS[x]  - pSW[x] - pSE[x] - pTS[x] - pBS[x] );
               velZ[x] = invRho * ( velZ_trm + pTN[x] + pTE[x] - pB[x]  - pBN[x] - pBS[x] - pBW[x] - pBE[x] );

               t1x2[x] = t1x2_0 * rho;
               t2x2[x] = t2x2_0 * rho;
               fac1[x] = t1x2_0 * rho * inv2csq2;
               fac2[x] = t2x2_0 * rho * inv2csq2;

               feq_common[x] = real_t(1.0) - real_t(1.5) * ( velX[x] * velX[x] + velY[x] * velY[x] + velZ[x] * velZ[x] );

               pC[x] = pC[x] * (real_t(1.0) - lambda_e) + lambda_e * t0_0 * rho * feq_common[x];
            }

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t velXPY = velX[x] + velY[x];
               const real_t  sym_NE_SW = lambda_e_scaled * ( pNE[x] + pSW[x] - fac2[x] * velXPY * velXPY - t2x2[x] * feq_common[x] );
               const real_t asym_NE_SW = lambda_d_scaled * ( pNE[x] - pSW[x] - real_t(3.0) * t2x2[x] * velXPY );

               pNE[x] = pNE[x] - sym_NE_SW - asym_NE_SW;
               pSW[x] = pSW[x] - sym_NE_SW + asym_NE_SW;
            }

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t velXMY = velX[x] - velY[x];
               const real_t  sym_SE_NW = lambda_e_scaled * ( pSE[x] + pNW[x] - fac2[x] * velXMY * velXMY - t2x2[x] * feq_common[x] );
               const real_t asym_SE_NW = lambda_d_scaled * ( pSE[x] - pNW[x] - real_t(3.0) * t2x2[x] * velXMY );

               pSE[x] = pSE[x] - sym_SE_NW - asym_SE_NW;
               pNW[x] = pNW[x] - sym_SE_NW + asym_SE_NW;
            }

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t velXPZ = velX[x] + velZ[x];
               const real_t  sym_TE_BW = lambda_e_scaled * ( pTE[x] + pBW[x] - fac2[x] * velXPZ * velXPZ - t2x2[x] * feq_common[x] );
               const real_t asym_TE_BW = lambda_d_scaled * ( pTE[x] - pBW[x] - real_t(3.0) * t2x2[x] * velXPZ );

               pTE[x] = pTE[x] - sym_TE_BW - asym_TE_BW;
               pBW[x] = pBW[x] - sym_TE_BW + asym_TE_BW;
            }

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t velXMZ = velX[x] - velZ[x];
               const real_t  sym_BE_TW = lambda_e_scaled * ( pBE[x] + pTW[x] - fac2[x] * velXMZ * velXMZ - t2x2[x] * feq_common[x] );
               const real_t asym_BE_TW = lambda_d_scaled * ( pBE[x] - pTW[x] - real_t(3.0) * t2x2[x] * velXMZ );

               pBE[x] = pBE[x] - sym_BE_TW - asym_BE_TW;
               pTW[x] = pTW[x] - sym_BE_TW + asym_BE_TW;
            }

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t velYPZ = velY[x] + velZ[x];
               const real_t  sym_TN_BS = lambda_e_scaled * ( pTN[x] + pBS[x] - fac2[x] * velYPZ * velYPZ - t2x2[x] * feq_common[x] );
               const real_t asym_TN_BS = lambda_d_scaled * ( pTN[x] - pBS[x] - real_t(3.0) * t2x2[x] * velYPZ );

               pTN[x] = pTN[x] - sym_TN_BS - asym_TN_BS;
               pBS[x] = pBS[x] - sym_TN_BS + asym_TN_BS;
            }

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t velYMZ = velY[x] - velZ[x];
               const real_t  sym_BN_TS = lambda_e_scaled * ( pBN[x] + pTS[x] - fac2[x] * velYMZ * velYMZ - t2x2[x] * feq_common[x] );
               const real_t asym_BN_TS = lambda_d_scaled * ( pBN[x] - pTS[x] - real_t(3.0) * t2x2[x] * velYMZ );

               pBN[x] = pBN[x] - sym_BN_TS - asym_BN_TS;
               pTS[x] = pTS[x] - sym_BN_TS + asym_BN_TS;
            }

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t  sym_N_S = lambda_e_scaled * ( pN[x] + pS[x] - fac1[x] * velY[x] * velY[x] - t1x2[x] * feq_common[x] );
               const real_t asym_N_S = lambda_d_scaled * ( pN[x] - pS[x] - real_t(3.0) * t1x2[x] * velY[x] );

               pN[x] = pN[x] - sym_N_S - asym_N_S;
               pS[x] = pS[x] - sym_N_S + asym_N_S;
            }

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t  sym_E_W = lambda_e_scaled * ( pE[x] + pW[x] - fac1[x] * velX[x] * velX[x] - t1x2[x] * feq_common[x] );
               const real_t asym_E_W = lambda_d_scaled * ( pE[x] - pW[x] - real_t(3.0) * t1x2[x] * velX[x] );

               pE[x] = pE[x] - sym_E_W - asym_E_W;
               pW[x] = pW[x] - sym_E_W + asym_E_W;
            }

            #pragma ibm iterations(100)
            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t  sym_T_B = lambda_e_scaled * ( pT[x] + pB[x]  - fac1[x] * velZ[x] * velZ[x] - t1x2[x] * feq_common[x] );
               const real_t asym_T_B = lambda_d_scaled * ( pT[x] - pB[x] - real_t(3.0) * t1x2[x] * velZ[x] );

               pT[x] = pT[x] - sym_T_B - asym_T_B;
               pB[x] = pB[x] - sym_T_B + asym_T_B;
            }
         }
      }
   }
   else // ==> src->layout() == field::zyxf
   {
      #ifdef _OPENMP
      const int izSize = int_c( zSize );
      #pragma omp for schedule(static)
      for( int iz = 0; iz < izSize; ++iz ) {
         cell_idx_t z = cell_idx_c( iz );
      #else
      for( cell_idx_t z = 0; z < zSize; ++z ) {
      #endif
         for( cell_idx_t y = 0; y != ySize; ++y )
         {
            using namespace stencil;

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_C  = src->get( x, y, z, Stencil::idx[C]  );
               const real_t dd_tmp_N  = src->get( x, y, z, Stencil::idx[N]  );
               const real_t dd_tmp_S  = src->get( x, y, z, Stencil::idx[S]  );
               const real_t dd_tmp_W  = src->get( x, y, z, Stencil::idx[W]  );
               const real_t dd_tmp_E  = src->get( x, y, z, Stencil::idx[E]  );
               const real_t dd_tmp_T  = src->get( x, y, z, Stencil::idx[T]  );
               const real_t dd_tmp_B  = src->get( x, y, z, Stencil::idx[B]  );
               const real_t dd_tmp_NW = src->get( x, y, z, Stencil::idx[NW] );
               const real_t dd_tmp_NE = src->get( x, y, z, Stencil::idx[NE] );
               const real_t dd_tmp_SW = src->get( x, y, z, Stencil::idx[SW] );
               const real_t dd_tmp_SE = src->get( x, y, z, Stencil::idx[SE] );
               const real_t dd_tmp_TN = src->get( x, y, z, Stencil::idx[TN] );
               const real_t dd_tmp_TS = src->get( x, y, z, Stencil::idx[TS] );
               const real_t dd_tmp_TW = src->get( x, y, z, Stencil::idx[TW] );
               const real_t dd_tmp_TE = src->get( x, y, z, Stencil::idx[TE] );
               const real_t dd_tmp_BN = src->get( x, y, z, Stencil::idx[BN] );
               const real_t dd_tmp_BS = src->get( x, y, z, Stencil::idx[BS] );
               const real_t dd_tmp_BW = src->get( x, y, z, Stencil::idx[BW] );
               const real_t dd_tmp_BE = src->get( x, y, z, Stencil::idx[BE] );

               const real_t velX_trm = dd_tmp_E + dd_tmp_NE + dd_tmp_SE + dd_tmp_TE + dd_tmp_BE;
               const real_t velY_trm = dd_tmp_N + dd_tmp_NW + dd_tmp_TN + dd_tmp_BN;
               const real_t velZ_trm = dd_tmp_T + dd_tmp_TS + dd_tmp_TW;

               const real_t rho = dd_tmp_C + dd_tmp_S + dd_tmp_W + dd_tmp_B + dd_tmp_SW + dd_tmp_BS + dd_tmp_BW + velX_trm + velY_trm + velZ_trm;
               const real_t invRho = real_t(1.0) / rho;

               velX[x] = invRho * ( velX_trm - dd_tmp_W  - dd_tmp_NW - dd_tmp_SW - dd_tmp_TW - dd_tmp_BW );
               velY[x] = invRho * ( velY_trm + dd_tmp_NE - dd_tmp_S  - dd_tmp_SW - dd_tmp_SE - dd_tmp_TS - dd_tmp_BS );
               velZ[x] = invRho * ( velZ_trm + dd_tmp_TN + dd_tmp_TE - dd_tmp_B  - dd_tmp_BN - dd_tmp_BS - dd_tmp_BW - dd_tmp_BE );

               t1x2[x] = t1x2_0 * rho;
               t2x2[x] = t2x2_0 * rho;
               fac1[x] = t1x2_0 * rho * inv2csq2;
               fac2[x] = t2x2_0 * rho * inv2csq2;

               feq_common[x] = real_t(1.0) - real_t(1.5) * ( velX[x] * velX[x] + velY[x] * velY[x] + velZ[x] * velZ[x] );

               src->get( x, y, z, Stencil::idx[C] ) = dd_tmp_C * (real_t(1.0) - lambda_e) + lambda_e * t0_0 * rho * feq_common[x];
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_NE = src->get( x, y, z, Stencil::idx[NE]);
               const real_t dd_tmp_SW = src->get( x, y, z, Stencil::idx[SW]);

               const real_t velXPY = velX[x] + velY[x];
               const real_t  sym_NE_SW = lambda_e_scaled * ( dd_tmp_NE + dd_tmp_SW - fac2[x] * velXPY * velXPY - t2x2[x] * feq_common[x] );
               const real_t asym_NE_SW = lambda_d_scaled * ( dd_tmp_NE - dd_tmp_SW - real_t(3.0) * t2x2[x] * velXPY );

               src->get( x, y, z, Stencil::idx[NE] ) = dd_tmp_NE - sym_NE_SW - asym_NE_SW;
               src->get( x, y, z, Stencil::idx[SW] ) = dd_tmp_SW - sym_NE_SW + asym_NE_SW;
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_SE = src->get( x, y, z, Stencil::idx[SE]);
               const real_t dd_tmp_NW = src->get( x, y, z, Stencil::idx[NW]);

               const real_t velXMY = velX[x] - velY[x];
               const real_t  sym_SE_NW = lambda_e_scaled * ( dd_tmp_SE + dd_tmp_NW - fac2[x] * velXMY * velXMY - t2x2[x] * feq_common[x] );
               const real_t asym_SE_NW = lambda_d_scaled * ( dd_tmp_SE - dd_tmp_NW - real_t(3.0) * t2x2[x] * velXMY );

               src->get( x, y, z, Stencil::idx[SE] ) = dd_tmp_SE - sym_SE_NW - asym_SE_NW;
               src->get( x, y, z, Stencil::idx[NW] ) = dd_tmp_NW - sym_SE_NW + asym_SE_NW;
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_TE = src->get( x, y, z, Stencil::idx[TE]);
               const real_t dd_tmp_BW = src->get( x, y, z, Stencil::idx[BW]);

               const real_t velXPZ = velX[x] + velZ[x];
               const real_t  sym_TE_BW = lambda_e_scaled * ( dd_tmp_TE + dd_tmp_BW - fac2[x] * velXPZ * velXPZ - t2x2[x] * feq_common[x] );
               const real_t asym_TE_BW = lambda_d_scaled * ( dd_tmp_TE - dd_tmp_BW - real_t(3.0) * t2x2[x] * velXPZ );

               src->get( x, y, z, Stencil::idx[TE] ) = dd_tmp_TE - sym_TE_BW - asym_TE_BW;
               src->get( x, y, z, Stencil::idx[BW] ) = dd_tmp_BW - sym_TE_BW + asym_TE_BW;
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_BE = src->get( x, y, z, Stencil::idx[BE]);
               const real_t dd_tmp_TW = src->get( x, y, z, Stencil::idx[TW]);

               const real_t velXMZ = velX[x] - velZ[x];
               const real_t  sym_BE_TW = lambda_e_scaled * ( dd_tmp_BE + dd_tmp_TW - fac2[x] * velXMZ * velXMZ - t2x2[x] * feq_common[x] );
               const real_t asym_BE_TW = lambda_d_scaled * ( dd_tmp_BE - dd_tmp_TW - real_t(3.0) * t2x2[x] * velXMZ );

               src->get( x, y, z, Stencil::idx[BE] ) = dd_tmp_BE - sym_BE_TW - asym_BE_TW;
               src->get( x, y, z, Stencil::idx[TW] ) = dd_tmp_TW - sym_BE_TW + asym_BE_TW;
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_TN = src->get( x, y, z, Stencil::idx[TN]);
               const real_t dd_tmp_BS = src->get( x, y, z, Stencil::idx[BS]);

               const real_t velYPZ = velY[x] + velZ[x];
               const real_t  sym_TN_BS = lambda_e_scaled * ( dd_tmp_TN + dd_tmp_BS - fac2[x] * velYPZ * velYPZ - t2x2[x] * feq_common[x] );
               const real_t asym_TN_BS = lambda_d_scaled * ( dd_tmp_TN - dd_tmp_BS - real_t(3.0) * t2x2[x] * velYPZ );

               src->get( x, y, z, Stencil::idx[TN] ) = dd_tmp_TN - sym_TN_BS - asym_TN_BS;
               src->get( x, y, z, Stencil::idx[BS] ) = dd_tmp_BS - sym_TN_BS + asym_TN_BS;
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_BN = src->get( x, y, z, Stencil::idx[BN]);
               const real_t dd_tmp_TS = src->get( x, y, z, Stencil::idx[TS]);

               const real_t velYMZ = velY[x] - velZ[x];
               const real_t  sym_BN_TS = lambda_e_scaled * ( dd_tmp_BN + dd_tmp_TS - fac2[x] * velYMZ * velYMZ - t2x2[x] * feq_common[x] );
               const real_t asym_BN_TS = lambda_d_scaled * ( dd_tmp_BN - dd_tmp_TS - real_t(3.0) * t2x2[x] * velYMZ );

               src->get( x, y, z, Stencil::idx[BN] ) = dd_tmp_BN - sym_BN_TS - asym_BN_TS;
               src->get( x, y, z, Stencil::idx[TS] ) = dd_tmp_TS - sym_BN_TS + asym_BN_TS;
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_N  = src->get( x, y, z, Stencil::idx[N]);
               const real_t dd_tmp_S  = src->get( x, y, z, Stencil::idx[S]);

               const real_t  sym_N_S = lambda_e_scaled * ( dd_tmp_N + dd_tmp_S - fac1[x] * velY[x] * velY[x] - t1x2[x] * feq_common[x] );
               const real_t asym_N_S = lambda_d_scaled * ( dd_tmp_N - dd_tmp_S - real_t(3.0) * t1x2[x] * velY[x] );

               src->get( x, y, z, Stencil::idx[N] ) = dd_tmp_N - sym_N_S - asym_N_S;
               src->get( x, y, z, Stencil::idx[S] ) = dd_tmp_S - sym_N_S + asym_N_S;
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_E  = src->get( x, y, z, Stencil::idx[E]);
               const real_t dd_tmp_W  = src->get( x, y, z, Stencil::idx[W]);

               const real_t  sym_E_W = lambda_e_scaled * ( dd_tmp_E + dd_tmp_W - fac1[x] * velX[x] * velX[x] - t1x2[x] * feq_common[x] );
               const real_t asym_E_W = lambda_d_scaled * ( dd_tmp_E - dd_tmp_W - real_t(3.0) * t1x2[x] * velX[x] );

               src->get( x, y, z, Stencil::idx[E] ) = dd_tmp_E - sym_E_W - asym_E_W;
               src->get( x, y, z, Stencil::idx[W] ) = dd_tmp_W - sym_E_W + asym_E_W;
            }

            for( cell_idx_t x = 0; x != xSize; ++x )
            {
               const real_t dd_tmp_T  = src->get( x, y, z, Stencil::idx[T]);
               const real_t dd_tmp_B  = src->get( x, y, z, Stencil::idx[B]);

               const real_t  sym_T_B = lambda_e_scaled * ( dd_tmp_T + dd_tmp_B  - fac1[x] * velZ[x] * velZ[x] - t1x2[x] * feq_common[x] );
               const real_t asym_T_B = lambda_d_scaled * ( dd_tmp_T - dd_tmp_B - real_t(3.0) * t1x2[x] * velZ[x] );

               src->get( x, y, z, Stencil::idx[T] ) = dd_tmp_T - sym_T_B - asym_T_B;
               src->get( x, y, z, Stencil::idx[B] ) = dd_tmp_B - sym_T_B + asym_T_B;
            }
         }
      }
   }

   delete[] velX;
   delete[] velY;
   delete[] velZ;
   delete[] t1x2;
   delete[] t2x2;
   delete[] fac1;
   delete[] fac2;
   delete[] feq_common;

#ifdef _OPENMP
   }
#endif
}



} // namespace lbm
} // namespace walberla

#endif
