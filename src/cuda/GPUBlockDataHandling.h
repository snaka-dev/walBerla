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
//! \file GPUBlockDataHandling.h
//! \ingroup cuda
//! \author Markus Holzer <markus.holzer@fau.de>
//
//======================================================================================================================

#pragma once

#include "blockforest/BlockDataHandling.h"
#include "blockforest/StructuredBlockForest.h"
#include "core/debug/CheckFunctions.h"
#include "core/math/Vector2.h"
#include "core/math/Vector3.h"
#include "field/FlagField.h"

#include <type_traits>


namespace walberla {
namespace cuda {



// still virtual, one must implement protected member functions 'allocate' and 'reallocate'
template< typename Field_T, bool Pseudo2D = false >
class GPUBlockDataHandling : public blockforest::BlockDataHandling< Field_T >
{
public:

   typedef typename Field_T::value_type Value_T;
   typedef std::function< void ( Field_T * field, IBlock * const block ) > InitializationFunction_T;

   virtual ~GPUBlockDataHandling() = default;

   void addInitializationFunction( const InitializationFunction_T & initFunction ) { initFunction_ = initFunction; }

   Field_T * initialize( IBlock * const block )
   {
      Field_T * field = allocate( block );
      return field;
   }

   inline void serialize( ) {};

   void serializeCoarseToFine( ) {};
   void serializeFineToCoarse( ) {};

   Field_T * deserialize( IBlock * const block ) { return reallocate( block ); }

   Field_T * deserializeCoarseToFine( Block * const block ) { return reallocate( block ); }
   Field_T * deserializeFineToCoarse( Block * const block ) { return reallocate( block ); }   
   
   void deserialize( ) {};

   void deserializeCoarseToFine( ) {};
   void deserializeFineToCoarse( ) {};

protected:

   /// must be thread-safe !
   virtual Field_T *   allocate( IBlock * const block ) = 0; // used in 'initialize'
   /// must be thread-safe !
   virtual Field_T * reallocate( IBlock * const block ) = 0; // used in all deserialize member functions

   template< typename T > struct Merge
   { static T result( const T & value ) { return Pseudo2D ? static_cast<T>( value / numeric_cast<T>(4) ) : static_cast<T>( value / numeric_cast<T>(8) ); } };

   template< typename T > struct Merge< Vector2<T> >
   { static Vector2<T> result( const Vector2<T> & value ) { return Pseudo2D ? (value / numeric_cast<T>(4)) : (value / numeric_cast<T>(8)); } };

   template< typename T > struct Merge< Vector3<T> >
   { static Vector3<T> result( const Vector3<T> & value ) { return Pseudo2D ? (value / numeric_cast<T>(4)) : (value / numeric_cast<T>(8)); } };

   void sizeCheck( const uint_t xSize, const uint_t ySize, const uint_t zSize )
   {
      WALBERLA_CHECK( (xSize & uint_t(1)) == uint_t(0), "The x-size of your field must be divisible by 2." );
      WALBERLA_CHECK( (ySize & uint_t(1)) == uint_t(0), "The y-size of your field must be divisible by 2." );
      if( Pseudo2D )
      { WALBERLA_CHECK( zSize == uint_t(1), "The z-size of your field must be equal to 1 (pseudo 2D mode)." ); }
      else
      { WALBERLA_CHECK( (zSize & uint_t(1)) == uint_t(0), "The z-size of your field must be divisible by 2." ); }
   }
   
   InitializationFunction_T initFunction_;

}; // class GPUBlockDataHandling

} // namespace field
} // namespace walberla
