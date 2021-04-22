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
//! \file
//! \author Igor Ostanin <i.ostanin@skoltech.ru>
//! \author Grigorii Drozdov <drozd013@umn.edu>
//! \author Sebastian Eibl <sebastian.eibl@fau.de>
//
//======================================================================================================================

#pragma once

#include <mesa_pd/common/ParticleFunctions.h>
#include <mesa_pd/data/DataTypes.h>
#include <mesa_pd/data/IAccessor.h>

#include <core/math/Angles.h>
#include <core/math/Constants.h>
#include <core/logging/Logging.h>

#include <vector>
#include <iomanip>

namespace walberla {
namespace mesa_pd {
namespace kernel {
namespace cnt {

/**
 * new anisotropic vdW contact
 */
class AnisotropicNewVDWContact
{
public:
   template<typename Accessor>
   void operator()(const size_t p_idx1,
                   const size_t p_idx2,
                   Accessor &ac);

   static constexpr real_t alf_ = 9.5_r;
   static constexpr real_t bet_ = 4.0_r;
   static constexpr real_t Cg_ = 90_r;
   static constexpr real_t del_ = -7.5_r;

   static constexpr size_t M = 21;

   static constexpr std::array< real_t, M > aA = {
      -191727.4839426546_r,   -371910.75942337443_r, -339251.7492620516_r, -290865.4561642854_r,
                       -234144.88122029652_r,  -176696.12760688397_r, -124742.12768384402_r, -82161.34671475281_r,
                       -50314.71182382854_r,   -28523.965679291196_r, -14888.017859788077_r, -7105.05919507198_r,
                       -3072.916729834469_r,   -1190.6127633797219_r, -406.9466114882997_r,  -120.12457898322344_r,
                       -29.700204039001516_r,  -5.867890961396333_r,  -0.8555072105106888_r, -0.07861570142351418_r,
                       -0.002966294228854819_r };
   static constexpr std::array< real_t, M > bA = {
      3.2088720077169565e6_r,  -3.2847658083596528e6_r, -3.1670894942180435e6_r, 6.926816667566258e6_r,
                       -4.56572526926024e6_r,  -1.8110081402744525e6_r, 6.815072405354291e6_r,   -6.581146873836791e6_r,
                       1.5536562641377014e6_r, 4.675280594755341e6_r,   -8.598536244610775e6_r,  9.024007030926665e6_r,
                       -6.985760263169681e6_r, 4.263132144718523e6_r,   -2.0936417014329613e6_r, 827318.3856828389_r,
                       -259020.37396180714_r,  62181.529468855704_r,    -10778.961199478841_r,   1200.1703972475002_r,
                       -64.0650502845509_r };
   static constexpr std::array< real_t, M > aB = {
      1.1410668710057335e6_r, 2.224384820605823e6_r, 2.0592863531518714e6_r, 1.8095080312026043e6_r,
                       1.5074251731701875e6_r, 1.188569925584079e6_r, 885084.7507304312_r,    620766.3917102363_r,
                       408686.0719918859_r,    251524.59513471829_r,  143984.66948507103_r,   76193.11216199506_r,
                       36986.74017282174_r,    16311.346712260749_r,  6453.233405301696_r,    2252.0573797063016_r,
                       677.0684067365061_r,    169.29950334766284_r,  33.2463154561197_r,     4.601254819561541_r,
                       0.3411306585741447_r };
   static constexpr std::array< real_t, M > bB = {
      1.5129361849136291e7_r,  -1.5378549855421165e7_r, -1.514133063600646e7_r,  3.263386057569147e7_r,
                       -2.1065431505164593e7_r, -9.088670359272573e6_r,  3.2116674880978897e7_r,  -3.0199664147810895e7_r,
                       6.2263944378619855e6_r,  2.2457316542130932e7_r,  -3.9674070148773186e7_r, 4.057912896618676e7_r,
                       -3.059350303152226e7_r,  1.810077205947395e7_r,   -8.554358959725464e6_r,  3.216534735737917e6_r,
                       -941825.2601902693_r,    205579.23892303705_r,    -30801.033934519535_r,   2661.333228648137_r,
                       -79.4553168412277_r };
   static constexpr std::array< real_t, M > aR = {
      2.611646016044291e7_r,  5.078047807870879e7_r,  4.665284854260373e7_r,  4.048148790394097e7_r,
                       3.3146389102277704e7_r, 2.5576684311130513e7_r, 1.8565558980651785e7_r, 1.2648089493946007e7_r,
                       8.063414365506318e6_r,  4.792706173483312e6_r,  2.643524343410147e6_r,  1.3451214360724073e6_r,
                       626667.550675059_r,     264701.2397312482_r,    100063.57481906567_r,   33257.77686751788_r,
                       9476.730776731225_r,    2229.0341880801734_r,   406.8187300697395_r,    51.30987450669856_r,
                       3.3545558972293845_r };
   static constexpr std::array< real_t, M > bR = {
      -7.325237436038056e7_r, 7.527996554523401e7_r,   7.173613688567044e7_r,   -1.582070701261235e8_r,
                       1.0548122803947036e8_r, 3.988318259645124e7_r,   -1.55669924013528e8_r,  1.5248766416875613e8_r,
                       -3.835046594542539e7_r, -1.0574716200460382e8_r, 1.9868563429213738e8_r, -2.1117587031365895e8_r,
                       1.6546969602932808e8_r, -1.023119914833957e8_r,  5.099951418492136e7_r,  -2.050607669664799e7_r,
                       6.554626662945287e6_r,  -1.6138986152148629e6_r, 288824.27923296514_r,   -33529.328462671256_r,
                       1896.4034102788603_r };

   /// vdW interaction radius w/r to inertial segment radius
   static constexpr real_t CutoffFactor     = 4_r;
   /// CNT radius
   static constexpr real_t R_CNT = 6.78_r;
   static constexpr real_t R_ = R_CNT;
   static constexpr real_t r_cut_ = CutoffFactor * R_;

   auto isParallel() const {return isParallel_;}
   auto getLastEnergy() const {return energy_;}
private:
   real_t energy_; ///< total potential
   bool isParallel_;
};

template<typename Accessor>
inline
void AnisotropicNewVDWContact::operator()(const size_t p_idx1,
                                       const size_t p_idx2,
                                       Accessor &ac)
{
   isParallel_ = false;
   energy_ = 0_r;
   //===Adaptation of PFC5 vdW contact model implementation====

   // Getting the orientations of segments
   Vec3 b1 = ac.getRotation(p_idx1).getMatrix() * Vec3(1.0, 0.0, 0.0); ///< ball 1 axial direction
   Vec3 b2 = ac.getRotation(p_idx2).getMatrix() * Vec3(1.0, 0.0, 0.0); ///< ball 2 axial direction

   // Distance between segments

   Vec3 n = ac.getPosition(p_idx2) - ac.getPosition(p_idx1); ///< contact normal
   auto L = n.length();
   n *= (1_r/L);

   //WALBERLA_LOG_DEVEL( "Normal: n = " << n );
   //WALBERLA_LOG_DEVEL( "Orientation of seg 1: b1 = " << b1);
   //WALBERLA_LOG_DEVEL( "Orientation of seg 2: b2 = " << b2);
   //WALBERLA_LOG_DEVEL( "Length of rad vect: L = " << L );

   constexpr real_t TOL = 10e-8_r;

   //---------------------
   // NORMALS CALCULATION
   //---------------------
   // names convention:
   // c1 - contact 1-2 normal
   // b1 - ball 1 axial direction
   // b2 - ball 2 axial direction
   // b3 - neutral direction
   // g - aligning torque direction
   // d - neutral plane normal direction
   // s - shear force direction

   // angle gamma - angle between two axial directions
   auto cos_gamma = b1 * b2;

   // if the angle between two axal directions is blunt, then inverce b2
   if (cos_gamma < 0_r)
   {
      b2 = -b2;
      cos_gamma = -cos_gamma;
   }
   //WALBERLA_LOG_DEVEL("Orientation of seg 2 again: b2 = " << b2);
   // check that cosine belongs [-1,1]
   cos_gamma = std::min(1.0_r, cos_gamma);
   cos_gamma = std::max(-1.0_r, cos_gamma);
   if (L < 20_r && L > 16_r)
   {
      const auto gamma = acos(cos_gamma);
      if (gamma < math::degToRad(10_r) || gamma > math::degToRad(170_r))
         isParallel_ = true;
   }
   //WALBERLA_LOG_DEVEL( "cos_gamma: = " << cos_gamma );

   // calculate functions of double argument
   auto sin_gamma = std::sqrt(1.0_r - cos_gamma * cos_gamma);
   auto cos_2gamma = cos_gamma * cos_gamma - sin_gamma * sin_gamma;
   auto sin_2gamma = 2.0_r * sin_gamma * cos_gamma;

   //WALBERLA_LOG_DEVEL( "sin_gamma: = " << sin_gamma );
   //WALBERLA_LOG_DEVEL( "cos_2gamma: = " << cos_2gamma );
   //WALBERLA_LOG_DEVEL( "sin_2gamma: = " << sin_2gamma );

   // g - direction of the aligning torques - b1 X b2
   Vec3 g(0.0, 0.0, 0.0);
   if (sin_gamma > TOL)
   {
      g = b1 % b2;
      g = g * (1.0_r / g.length());
   }
   //WALBERLA_LOG_DEVEL( "Aligning moment direction: g = " << g );

   // b3 - vector defining the neutral plane ( plane of shear forces )
   Vec3 b3 = b1 + b2;
   b3 = b3 * (1.0_r / b3.length());
   //WALBERLA_LOG_DEVEL( "Neutral plane defined by b3 = " << b3 );

   // angle theta - angle between b3 and c1
   auto cos_theta = b3 * n;
   // check that cosine belongs [-1,1]
   cos_theta = std::min(1.0_r, cos_theta);
   cos_theta = std::max(-1.0_r, cos_theta);
   //WALBERLA_LOG_DEVEL("cos_theta: = " << cos_theta);
   const auto theta = acos(cos_theta);
   //WALBERLA_LOG_DEVEL("theta: = " << theta);

   // calculation of shear force direction
   Vec3 s(0.0, 0.0, 0.0);
   Vec3 d(0.0, 0.0, 0.0);

   if ((cos_theta > -1.0 + TOL) || (cos_theta < 1.0 - TOL))
      d = n % b3;
   s = n % d;
   s = s * (1.0_r / s.length());

   //WALBERLA_LOG_DEVEL( "Shear force direction: = " << s );

   //--------------------------------
   // NORMALS CALCULATION - END
   //--------------------------------

   // Fast calculation of trigonometric functions ( Chebyshev formulas )
   real_t coss[M], sinn[M];
   real_t sin_theta = std::sqrt(1.0_r - cos_theta * cos_theta);

   coss[0] = 1_r;
   sinn[0] = 0_r;
   coss[1] = cos_theta * cos_theta - sin_theta * sin_theta;
   sinn[1] = 2_r * sin_theta * cos_theta;
   for (size_t i = 1; i < M - 1; i++)
   {
      coss[i + 1] = coss[i] * coss[1] - sinn[i] * sinn[1];
      sinn[i + 1] = sinn[i] * coss[1] + sinn[1] * coss[i];
   }

   // Adjustment w/r to O
   real_t A = 0_r, A_O = 0_r, B = 0_r, B_O = 0_r, R = 0_r, R_O = 0_r;
   real_t theta_0 = 205._r / 500._r * math::pi / 2._r;
   if ((theta > theta_0) && (theta <= math::pi - theta_0))
   {
      for (size_t i = 0; i < M; i++)
      {
         A += aA[i] * coss[i];
         A_O -= 2_r * real_t(i) * aA[i] * sinn[i];
         B += aB[i] * coss[i];
         B_O -= 2_r * real_t(i) * aB[i] * sinn[i];
         R += aR[i] * coss[i];
         R_O -= 2_r * real_t(i) * aR[i] * sinn[i];
      }
   }
   else
   {
      for (size_t i = 0; i < M; i++)
      {
         A += bA[i] * coss[i];
         A_O -= 2_r * real_t(i) * bA[i] * sinn[i];
         B += bB[i] * coss[i];
         B_O -= 2_r * real_t(i) * bB[i] * sinn[i];
         R += bR[i] * coss[i];
         R_O -= 2_r * real_t(i) * bR[i] * sinn[i];
      }
   }

   //WALBERLA_LOG_DEVEL( "TH: = " << TH );
   //WALBERLA_LOG_DEVEL( "TH_L: = " << TH_L );
   //WALBERLA_LOG_DEVEL( "TH_O: = " << TH_O );

   //------------------------------------------------------------------
   // THIS BLOCK IMPLEMENTS IF THE DISTANCE L IS WITHIN WORKING RANGE
   //------------------------------------------------------------------

   if ((L < 2_r * r_cut_) && (L > R * 2_r))
   {
      //-----Constants that appear in the potential--------------------------
      // This set of constants is described in our paper.
      //---------------------------------------------------------------------

      //-----Function D and its derivatives-----------------------
      real_t D   = L / R - 2._r;
      real_t D_L = 1._r / R;
      real_t D_O = -(L * R_O) / (R * R);
      //-----------------------------------------------------------

      //WALBERLA_LOG_DEVEL( "D: = " << D );
      //WALBERLA_LOG_DEVEL( "D_L: = " << D_L );
      //WALBERLA_LOG_DEVEL( "D_O: = " << D_O );

      //----Function Vc and its derivatives---------------------------------------
      const real_t DpowAlpha0 = std::pow(D, -(alf_));
      const real_t DpowAlpha1 = std::pow(D, -(alf_ + 1));
      const real_t DpowBeta0 = std::pow(D, -(bet_));
      const real_t DpowBeta1 = std::pow(D, -(bet_ + 1));
      real_t Vc = A * DpowAlpha0 - B * DpowBeta0;
      real_t Vc_L = (-alf_ * A * DpowAlpha1 + bet_ * B * DpowBeta1) * D_L;
      real_t Vc_O = A_O * DpowAlpha0 - B_O * DpowBeta0 + (-alf_ * A * DpowAlpha1 + bet_ * B * DpowBeta1) * D_O;
      //--------------------------------------------------------------------------

      //WALBERLA_LOG_DEVEL( "VC = " << Vc );
      //WALBERLA_LOG_DEVEL( "VC_L = " << Vc_L );
      //WALBERLA_LOG_DEVEL( "VC_O = " << Vc_O );

      // Cutoff for u adjustment
      real_t W_u = 1_r;
      real_t W_u_L = 0_r;

      // Cubic cutoff function 3T->4T (hardcoded since we do not need to mess w these parameters)
      constexpr auto Q1_ = -80.0_r;
      constexpr auto Q2_ = 288.0_r;
      constexpr auto Q3_ = -336.0_r;
      constexpr auto Q4_ = 128.0_r;
      const real_t rcut2inv = 1_r / (2.0_r * r_cut_);
      real_t nd = L * rcut2inv;
      if ((nd > 0.75_r) && (nd < 1.0_r))
      {
         W_u = Q1_ + Q2_ * nd + Q3_ * nd * nd + Q4_ * nd * nd * nd;
         W_u_L = (Q2_ + 2.0_r * Q3_ * nd + 3.0_r * Q4_ * nd * nd) * rcut2inv;
      }
      //--------------------------------------------------------------------------

      //WALBERLA_LOG_DEVEL( "W_u = " << W_u );
      //WALBERLA_LOG_DEVEL( "W_u_L = " << W_u_L );

      // Cutoff for gamma adjustment

      real_t W_ga, W_ga_L;
      if (L / R_ > 2.75_r)
      {
         W_ga = Cg_ * std::pow((L / R_), del_);
         W_ga_L = ((del_ * Cg_) / R_) * std::pow((L / R_), del_ - 1_r);
      } else
      {
         W_ga = Cg_ * std::pow((2.75_r), del_);
         W_ga_L = 0;
      }

      //WALBERLA_LOG_DEVEL( "W_ga = " << W_ga );
      //WALBERLA_LOG_DEVEL( "W_ga_L = " << W_ga_L );

      real_t GA = 1_r;
      real_t GA_L = 0_r;
      real_t GA_G = 0_r;

      if (std::abs(sin_gamma) > TOL)
      {
         GA = 1_r + W_ga * (1_r - cos_2gamma);
         GA_L = W_ga_L * (1_r - cos_2gamma);
         GA_G = 2_r * W_ga * sin_2gamma;
      }

      //----Forces and torque-----------------------
      real_t FL = -GA_L * W_u * Vc - GA * W_u_L * Vc - GA * W_u * Vc_L;
      real_t FO = -(1_r / L) * GA * W_u * Vc_O;
      real_t MG = -GA_G * W_u * Vc;

      //WALBERLA_LOG_DEVEL( "FL = " << FL );
      //WALBERLA_LOG_DEVEL( "FO = " << FO );
      //WALBERLA_LOG_DEVEL( "MG = " << MG );

      Vec3 force = FL * n + FO * s;
      Vec3 moment = MG * g;

      //WALBERLA_LOG_DEVEL("Contact force: = " << force.length());
      //WALBERLA_LOG_DEVEL("Contact moment: = " << moment);

      if (force.length() > 100._r)
      {
         force = force * (100._r / force.length());
         addForceAtomic(p_idx1, ac, -force);
         addForceAtomic(p_idx2, ac, force);
      }
      else
      {
         // Potential energy
         energy_ = GA * W_u * Vc;

         addForceAtomic(p_idx1, ac, -force);
         addForceAtomic(p_idx2, ac, force);
         addTorqueAtomic(p_idx1, ac, -moment);
         addTorqueAtomic(p_idx2, ac, moment);
      }

      // WALBERLA_LOG_DEVEL( "U_vdw = " << U );
   }
   else if (L <= R * 2.0)
   { // Small distance
      //WALBERLA_LOG_DEVEL( "Small distance");
      real_t F = -1_r;
      if (L < TOL)
      {
         addForceAtomic(p_idx1, ac, F * Vec3(1, 0, 0));
         addForceAtomic(p_idx2, ac, -F * Vec3(1, 0, 0));
      }
      else
      {
         addForceAtomic(p_idx1, ac, F * n);
         addForceAtomic(p_idx2, ac, -F * n);
      }
   }
}

} //namespace cnt
} //namespace kernel
} //namespace mesa_pd
} //namespace walberla