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
//! \file Any.h
//! \ingroup core
//! \author Michael Kuron <mkuron@icp.uni-stuttgart.de>
//
//======================================================================================================================

#pragma once

#include "waLBerlaDefinitions.h"

#ifndef WALBERLA_USE_STD_EXPERIMENTAL_ANY
#include <any>
#else
#undef _LIBCPP_WARN_ON_DEPRECATED_EXPERIMENTAL_HEADER
#include <experimental/any>
#endif



namespace walberla {

#ifndef WALBERLA_USE_STD_EXPERIMENTAL_ANY
using std::any;
using std::any_cast;
#else
using std::experimental::any;
using std::experimental::any_cast;
#endif

}
