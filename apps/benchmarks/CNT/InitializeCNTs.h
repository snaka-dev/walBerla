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

#include "FilmSpecimen.h"

#include "mesa_pd/data/ParticleStorage.h"
#include "mesa_pd/domain/IDomain.h"

namespace walberla{
namespace mesa_pd {

void make_element(const shared_ptr< data::ParticleStorage >& ps, int myRank, int segment, int tube, const Vec3& pos,
                  real_t theta, real_t phi, int64_t& numParticles);

void make_tube(const FilmSpecimen& spec, const shared_ptr< data::ParticleStorage >& ps, int myRank,
               const domain::IDomain& domain, int id, int n, const Vec3& t_pos, real_t theta, real_t phi,
               int64_t& numParticles);

void make_bundle(const FilmSpecimen& spec, const shared_ptr< data::ParticleStorage >& ps, int myRank,
                 const domain::IDomain& domain, int id, int side, int n, const Vec3& pos, real_t theta, real_t phi,
                 real_t alf, int64_t& numParticles);

int64_t generateCNTs(const FilmSpecimen& spec,
                     const shared_ptr<data::ParticleStorage>& ps,
                     const domain::IDomain& domain);

int64_t loadCNTs(const std::string& filename,
                 const shared_ptr<data::ParticleStorage>& ps,
                 const domain::IDomain& domain);

} //namespace mesa_pd
} //namespace walberla
