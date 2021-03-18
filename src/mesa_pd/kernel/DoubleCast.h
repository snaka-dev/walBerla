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
//! \author Sebastian Eibl <sebastian.eibl@fau.de>
//
//======================================================================================================================

//======================================================================================================================
//
//  THIS FILE IS GENERATED - PLEASE CHANGE THE TEMPLATE !!!
//
//======================================================================================================================

#pragma once

#include <mesa_pd/data/DataTypes.h>
#include <mesa_pd/data/IAccessor.h>
#include <mesa_pd/data/shape/BaseShape.h>
#include <mesa_pd/data/shape/Sphere.h>
#include <mesa_pd/data/shape/HalfSpace.h>
#include <mesa_pd/data/shape/CylindricalBoundary.h>
#include <mesa_pd/data/shape/Box.h>
#include <mesa_pd/data/shape/Ellipsoid.h>
#include <mesa_pd/data/shape/ConvexPolyhedron.h>

#include <core/Abort.h>
#include <core/debug/Debug.h>

namespace walberla {
namespace mesa_pd {
namespace kernel {

/**
 * This kernel requires the following particle accessor interface
 * \code
 * const BaseShape*& getShape(const size_t p_idx) const;
 *
 * \endcode
 * \ingroup mesa_pd_kernel
 */
class DoubleCast
{
public:
   template <typename Accessor, typename func, typename... Args>
   auto operator()( size_t idx, size_t idy, Accessor& ac, func& f, Args&&... args );
};

template <typename Accessor, typename func, typename... Args>
auto DoubleCast::operator()( size_t idx, size_t idy, Accessor& ac, func& f, Args&&... args )
{
   static_assert(std::is_base_of<data::IAccessor, Accessor>::value, "please provide a valid accessor");

   using namespace mesa_pd::data;

   switch (ac.getShape(idx)->getShapeType())
   {
      case Sphere::SHAPE_TYPE :
         switch (ac.getShape(idy)->getShapeType())
         {
            case Sphere::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<Sphere*>(ac.getShape(idx)),
                                                   *static_cast<Sphere*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            case HalfSpace::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<Sphere*>(ac.getShape(idx)),
                                                   *static_cast<HalfSpace*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            case CylindricalBoundary::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<Sphere*>(ac.getShape(idx)),
                                                   *static_cast<CylindricalBoundary*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            case Box::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<Sphere*>(ac.getShape(idx)),
                                                   *static_cast<Box*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            case Ellipsoid::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<Sphere*>(ac.getShape(idx)),
                                                   *static_cast<Ellipsoid*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            case ConvexPolyhedron::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<Sphere*>(ac.getShape(idx)),
                                                   *static_cast<ConvexPolyhedron*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            default : WALBERLA_ABORT("Shape type (" << ac.getShape(idy)->getShapeType() << ") could not be determined!");
         }
      case HalfSpace::SHAPE_TYPE :
         switch (ac.getShape(idy)->getShapeType())
         {
            case Sphere::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<HalfSpace*>(ac.getShape(idx)),
                                                   *static_cast<Sphere*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            case HalfSpace::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<HalfSpace*>(ac.getShape(idx)),
                                                   *static_cast<HalfSpace*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            case CylindricalBoundary::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<HalfSpace*>(ac.getShape(idx)),
                                                   *static_cast<CylindricalBoundary*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            case Box::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<HalfSpace*>(ac.getShape(idx)),
                                                   *static_cast<Box*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            case Ellipsoid::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<HalfSpace*>(ac.getShape(idx)),
                                                   *static_cast<Ellipsoid*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            case ConvexPolyhedron::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<HalfSpace*>(ac.getShape(idx)),
                                                   *static_cast<ConvexPolyhedron*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            default : WALBERLA_ABORT("Shape type (" << ac.getShape(idy)->getShapeType() << ") could not be determined!");
         }
      case CylindricalBoundary::SHAPE_TYPE :
         switch (ac.getShape(idy)->getShapeType())
         {
            case Sphere::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<CylindricalBoundary*>(ac.getShape(idx)),
                                                   *static_cast<Sphere*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            case HalfSpace::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<CylindricalBoundary*>(ac.getShape(idx)),
                                                   *static_cast<HalfSpace*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            case CylindricalBoundary::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<CylindricalBoundary*>(ac.getShape(idx)),
                                                   *static_cast<CylindricalBoundary*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            case Box::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<CylindricalBoundary*>(ac.getShape(idx)),
                                                   *static_cast<Box*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            case Ellipsoid::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<CylindricalBoundary*>(ac.getShape(idx)),
                                                   *static_cast<Ellipsoid*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            case ConvexPolyhedron::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<CylindricalBoundary*>(ac.getShape(idx)),
                                                   *static_cast<ConvexPolyhedron*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            default : WALBERLA_ABORT("Shape type (" << ac.getShape(idy)->getShapeType() << ") could not be determined!");
         }
      case Box::SHAPE_TYPE :
         switch (ac.getShape(idy)->getShapeType())
         {
            case Sphere::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<Box*>(ac.getShape(idx)),
                                                   *static_cast<Sphere*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            case HalfSpace::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<Box*>(ac.getShape(idx)),
                                                   *static_cast<HalfSpace*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            case CylindricalBoundary::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<Box*>(ac.getShape(idx)),
                                                   *static_cast<CylindricalBoundary*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            case Box::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<Box*>(ac.getShape(idx)),
                                                   *static_cast<Box*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            case Ellipsoid::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<Box*>(ac.getShape(idx)),
                                                   *static_cast<Ellipsoid*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            case ConvexPolyhedron::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<Box*>(ac.getShape(idx)),
                                                   *static_cast<ConvexPolyhedron*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            default : WALBERLA_ABORT("Shape type (" << ac.getShape(idy)->getShapeType() << ") could not be determined!");
         }
      case Ellipsoid::SHAPE_TYPE :
         switch (ac.getShape(idy)->getShapeType())
         {
            case Sphere::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<Ellipsoid*>(ac.getShape(idx)),
                                                   *static_cast<Sphere*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            case HalfSpace::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<Ellipsoid*>(ac.getShape(idx)),
                                                   *static_cast<HalfSpace*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            case CylindricalBoundary::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<Ellipsoid*>(ac.getShape(idx)),
                                                   *static_cast<CylindricalBoundary*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            case Box::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<Ellipsoid*>(ac.getShape(idx)),
                                                   *static_cast<Box*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            case Ellipsoid::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<Ellipsoid*>(ac.getShape(idx)),
                                                   *static_cast<Ellipsoid*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            case ConvexPolyhedron::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<Ellipsoid*>(ac.getShape(idx)),
                                                   *static_cast<ConvexPolyhedron*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            default : WALBERLA_ABORT("Shape type (" << ac.getShape(idy)->getShapeType() << ") could not be determined!");
         }
      case ConvexPolyhedron::SHAPE_TYPE :
         switch (ac.getShape(idy)->getShapeType())
         {
            case Sphere::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<ConvexPolyhedron*>(ac.getShape(idx)),
                                                   *static_cast<Sphere*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            case HalfSpace::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<ConvexPolyhedron*>(ac.getShape(idx)),
                                                   *static_cast<HalfSpace*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            case CylindricalBoundary::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<ConvexPolyhedron*>(ac.getShape(idx)),
                                                   *static_cast<CylindricalBoundary*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            case Box::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<ConvexPolyhedron*>(ac.getShape(idx)),
                                                   *static_cast<Box*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            case Ellipsoid::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<ConvexPolyhedron*>(ac.getShape(idx)),
                                                   *static_cast<Ellipsoid*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            case ConvexPolyhedron::SHAPE_TYPE : return f(idx,
                                                   idy,
                                                   *static_cast<ConvexPolyhedron*>(ac.getShape(idx)),
                                                   *static_cast<ConvexPolyhedron*>(ac.getShape(idy)),
                                                   std::forward<Args>(args)...);
            default : WALBERLA_ABORT("Shape type (" << ac.getShape(idy)->getShapeType() << ") could not be determined!");
         }
      default : WALBERLA_ABORT("Shape type (" << ac.getShape(idx)->getShapeType() << ") could not be determined!");
   }
}

} //namespace kernel
} //namespace mesa_pd
} //namespace walberla