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
//! \file SUID.h
//! \ingroup core
//! \author Florian Schornbaum <florian.schornbaum@fau.de>
//
//======================================================================================================================

#pragma once

#include "UID.h"
#include "UIDGenerators.h"
#include "UIDSet.h"


namespace walberla {
namespace uid {


namespace suidgenerator {
class S : public IndexGenerator< S, size_t >{}; ///< generator class for unified state/selection UIDs
}
using SUID = UID<suidgenerator::S>;           ///< unified state/selection UID


} // namespace uid

using uid::SUID;

} // namespace walberla
