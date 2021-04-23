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

#include "InitializeCNTs.h"
#include "TerminalColors.h"

#include "core/math/Constants.h"
#include "core/math/Random.h"
#include "core/mpi/MPIManager.h"
#include "mesa_pd/kernel/cnt/Parameters.h"

#include <cmath>

namespace walberla{
namespace mesa_pd {

void make_element(const shared_ptr< data::ParticleStorage >& ps, int myRank, int segment, int tube, const Vec3& pos,
                  real_t theta, real_t phi, int64_t& numParticles)
{
   data::Particle&& sp = *ps->create();
   sp.setPosition(pos);
   sp.setOwner(myRank);
   sp.setInteractionRadius(kernel::cnt::outer_radius);
   sp.setSegmentID(segment);
   sp.setClusterID(tube);
   sp.getRotationRef().rotate(Vec3(0_r, 1_r, 0_r), -0.5_r * math::pi + theta);
   sp.getRotationRef().rotate(Vec3(0_r, 0_r, 1_r), phi);
   numParticles++;
}

void make_tube(const FilmSpecimen& spec, const shared_ptr< data::ParticleStorage >& ps, int myRank,
               const domain::IDomain& domain, int id, int n, const Vec3& t_pos, real_t theta, real_t phi,
               int64_t& numParticles)
{
   auto CNT_length = 2_r * spec.spacing * real_c(n);
   Vec3 ax(std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta));
   for (int segment = 0; segment < n; segment++)
   {
      auto r   = -0.5_r * CNT_length + 2_r * spec.spacing * real_c(segment);
      Vec3 pos = t_pos + r * ax;

      if ((pos[0] > spec.sizeX)) pos[0] -= spec.sizeX;
      if ((pos[0] < 0)) pos[0] += spec.sizeX;
      if ((pos[1] > spec.sizeY)) pos[1] -= spec.sizeY;
      if ((pos[1] < 0)) pos[1] += spec.sizeY;
      if (spec.oopp)
      {
         if ((pos[2] > spec.sizeZ)) pos[2] -= spec.sizeZ;
         if ((pos[2] < 0)) pos[2] += spec.sizeZ;
      }

      if (domain.isContainedInProcessSubdomain(uint_c(myRank), pos))
         make_element(ps, myRank, segment, id, pos, theta, phi, numParticles);
   }
}

void make_bundle(const FilmSpecimen& spec, const shared_ptr< data::ParticleStorage >& ps, int myRank,
                 const domain::IDomain& domain, int id, int side, int n, const Vec3& pos, real_t theta, real_t phi,
                 real_t alf, int64_t& numParticles)
{
   real_t eq_dist = 17.15_r;
   real_t shift_x = 0.5_r * eq_dist;
   real_t shift_y = 0.5_r * std::sqrt(3_r) * eq_dist;
   int n_tu       = 2 * side - 1;
   int ii         = 0;
   Vec3 ax(-std::sin(phi), std::cos(phi), 0);
   Mat3 m(ax, theta);
   Vec3 ex        = m * Vec3(std::cos(alf), std::sin(alf), 0);
   Vec3 ey        = m * Vec3(-std::sin(alf), std::cos(alf), 0);
   Vec3 left_tube = pos - real_t(side - 1) * eq_dist * ex;
   Vec3 tube      = left_tube;
   for (int i = 0; i < n_tu; i++)
   {
      make_tube(spec, ps, myRank, domain, id * 100000 + ii, n, tube, theta, phi, numParticles);
      tube = tube + eq_dist * ex;
      ii++;
   }
   for (int i = 1; i < side; i++)
   {
      n_tu--;
      for (int k = -1; k < 3; k += 2)
      {
         tube = left_tube + i * (shift_x * ex + real_t(k) * shift_y * ey);
         for (int j = 0; j < n_tu; j++)
         {
            make_tube(spec, ps, myRank, domain, id * 100000 + ii, n, tube, theta, phi, numParticles);
            tube = tube + eq_dist * ex;
            ii++;
         }
      }
   }
}

int64_t generateCNTs(const FilmSpecimen& spec,
                     const shared_ptr<data::ParticleStorage>& ps,
                     const domain::IDomain& domain)
{
   auto myRank = mpi::MPIManager::instance()->rank();

   // Fixed random seed is necessary for coordinated generation on all MPI proc.
   auto rand0_1 = math::RealRandom<real_t>(static_cast<std::mt19937::result_type>(spec.seed));
   // Create an assembly of CNTs
   int64_t numParticles = 0;

   for (int tube = 0; tube < spec.numCNTs; tube++)
   {
      // This definition of theta provides uniform distribution of random orientations on a sphere when spec.OutOfPlane = 1.
      real_t theta = 0.5_r * math::pi * spec.min_OOP + (spec.max_OOP - spec.min_OOP) * std::acos(1_r - rand0_1());
      real_t phi = 2_r * math::pi * rand0_1();
      real_t pos_x = spec.sizeX * rand0_1();
      real_t pos_y = spec.sizeY * rand0_1();
      real_t pos_z = spec.sizeZ * rand0_1();
      make_tube(spec, ps, myRank, domain, tube, spec.numSegs, Vec3(pos_x, pos_y, pos_z), theta, phi, numParticles);
   }

   return numParticles;
}

int64_t loadCNTs(const std::string& filename,
                 const shared_ptr<data::ParticleStorage>& ps,
                 const domain::IDomain& domain)
{
   WALBERLA_LOG_INFO_ON_ROOT(GREEN << "Loading configuration (binary format): " << filename);
   const auto numProcesses = mpi::MPIManager::instance()->numProcesses();
   const auto rank = mpi::MPIManager::instance()->rank();
   //---------Generation of WaLBerla model---------------------------------------------------------------------------------------------------------------------------------
   // Note that walberla primitives are created below only on MPI process that is responsible
   // for this branch of blockforest

   int64_t numParticles = 0;

   for (auto i = 0; i < numProcesses; ++i)
   {
      WALBERLA_MPI_BARRIER();
      if (i != rank) continue; //bad practice but with the current file format we have to do this
      std::ifstream binfile;
      binfile.open(filename.c_str(), std::ios::in | std::ios::binary);
      size_t size;
      binfile.read((char *) &size, sizeof(size_t));
      std::cout << RED << "size read form binary file is" << size << RESET << std::endl;
      for (size_t id = 0; id < size; id++)
      {
         int64_t sID;
         int64_t cID;
         double x;
         double y;
         double z;
         double q0;
         double q1;
         double q2;
         double q3;
         double vx;
         double vy;
         double vz;
         double wx;
         double wy;
         double wz;
         double fx;
         double fy;
         double fz;
         double tx;
         double ty;
         double tz;
         binfile.read((char *) &sID, sizeof(int64_t));
         binfile.read((char *) &cID, sizeof(int64_t));
         binfile.read((char *) &x, sizeof(double));
         binfile.read((char *) &y, sizeof(double));
         binfile.read((char *) &z, sizeof(double));
         binfile.read((char *) &q0, sizeof(double));
         binfile.read((char *) &q1, sizeof(double));
         binfile.read((char *) &q2, sizeof(double));
         binfile.read((char *) &q3, sizeof(double));
         binfile.read((char *) &vx, sizeof(double));
         binfile.read((char *) &vy, sizeof(double));
         binfile.read((char *) &vz, sizeof(double));
         binfile.read((char *) &wx, sizeof(double));
         binfile.read((char *) &wy, sizeof(double));
         binfile.read((char *) &wz, sizeof(double));
         binfile.read((char *) &fx, sizeof(double));
         binfile.read((char *) &fy, sizeof(double));
         binfile.read((char *) &fz, sizeof(double));
         binfile.read((char *) &tx, sizeof(double));
         binfile.read((char *) &ty, sizeof(double));
         binfile.read((char *) &tz, sizeof(double));
         Vec3 pos;
         pos[0] = real_c(x);
         pos[1] = real_c(y);
         pos[2] = real_c(z);

         if (domain.isContainedInProcessSubdomain(uint_c(rank), pos))
         {
            data::Particle&& sp = *ps->create();
            sp.setPosition(pos);
            sp.setOwner(rank);
            sp.setInteractionRadius(kernel::cnt::outer_radius);
            sp.setSegmentID(sID);
            sp.setClusterID(cID);
            sp.setRotation(Rot3(Quaternion(q0, q1, q2, q3)));
            sp.setLinearVelocity(Vec3(real_c(vx), real_c(vy), real_c(vz)));
            sp.setAngularVelocity(Vec3(real_c(wx), real_c(wy), real_c(wz)));
            sp.setOldForce(Vec3(real_c(fx), real_c(fy), real_c(fz)));
            sp.setOldTorque(Vec3(real_c(tx), real_c(ty), real_c(tz)));
            numParticles++;
         }
      }
   }

   return numParticles;
}

} //namespace mesa_pd
} //namespace walberla

