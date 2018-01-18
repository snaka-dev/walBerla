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
//! \file BodySelectorFunctions.h
//! \ingroup pe_coupling
//! \author Christoph Rettinger <christoph.rettinger@fau.de>
//
//======================================================================================================================

#pragma once

#include "pe/Types.h"

namespace walberla {
namespace pe_coupling {

bool selectAllBodies(pe::BodyID /*bodyID*/)
{
   return true;
}

bool selectRegularBodies(pe::BodyID bodyID)
{
   return !bodyID->hasInfiniteMass() && !bodyID->isGlobal();
}

bool selectFixedBodies(pe::BodyID bodyID)
{
   return bodyID->hasInfiniteMass() && !bodyID->isGlobal();
}

bool selectGlobalBodies(pe::BodyID bodyID)
{
   return bodyID->isGlobal();
}
} // namespace pe_coupling
} // namespace walberla
