// Copyright (c) 2016 by Contributors
#include "integrity_tests.h"
#include <random>
#include "../common/sync.h"
#include "../data/simple_csr_source.h"
#include <xgboost\c_api.h>

namespace xgboost {

void XGIntegrityTests::DMatrixGroupSlices() {

  // static std::random_device _rd; // seed initialiser
  // static unsigned int _seed = _rd();
  unsigned int _seed = 0;
  std::mt19937 _rndGen(_seed); // Mersenne-Twister

  const int noOfGroups = 1000;
  const int noOfSliceGroups = 100;
  const int minGrpSize = 1, maxGrpSize = 10;

  // will create 3 structures:
  // 1) to pass directly into our initial dmatrix i.e., row & column:
  std::vector< std::vector<int> > rowStructure;
  // 2) to make creating slice easier i.e., group, row & column:
  std::vector< std::vector< std::vector<int> > > groupStructure;
  // 3) to hold group info
  std::vector<int> groupInfo;

  std::uniform_int_distribution<int> randGrps(minGrpSize, maxGrpSize);
  int id = 0;
  for (int i = 0; i < noOfGroups; i++) {
    int grpSize = randGrps(_rndGen);
    groupInfo.push_back(grpSize);
    std::vector< std::vector<int> > grp;
    for (int x = 0; x < grpSize; x++) {
      std::vector<int> row;
      row.push_back(id);  // will hold the column id in index 0 to make keeping track easier
      row.push_back(i);  // store group id number in index 1

      grp.push_back(row);
      rowStructure.push_back(row);

      id++;
    }
    groupStructure.push_back(grp);
  }

  // format data suitable for XGDMatrixCreateFromMat: 
  std::vector<float> data;
  for (unsigned i = 0; (i < rowStructure.size()); i++)
  {
    data.push_back(static_cast<float>(rowStructure[i][0]));
    data.push_back(static_cast<float>(rowStructure[i][1]));
  }

  DMatrixHandle dmAll;
  XGDMatrixCreateFromMat((float *)&data[0], rowStructure.size(), 2, -1, &dmAll);

  XGDMatrixSetGroup(dmAll, (const unsigned int*)&groupInfo[0], groupInfo.size());

  // get random groups of data:
  std::uniform_int_distribution<int> randG(0, noOfGroups - 1);
  std::vector<int> sliceGroups;
  while (sliceGroups.size() < noOfSliceGroups) {
    int randGrpIndex = randG(_rndGen);
    bool existsAlready = false;
    for (unsigned i = 0; i < sliceGroups.size(); i++) {
      if (sliceGroups[i] == randGrpIndex) {
        existsAlready = true;
        break;
      }
    }
    if (!existsAlready)
      sliceGroups.push_back(randGrpIndex);
  }

  // /*comment these out individually to test the Slice method against known issues*/
  // /*1) error: slice index is in wrong group:*/
  // int ix1 = -1;
  // int ix2 = -1;
  // for (int err = 0; err < noOfSliceGroups; err++) {
  //   if (groupStructure[sliceGroups[err]].size() > 1) {
  //     if (ix1 == -1)
  //       ix1 = sliceGroups[err];
  //     else if (ix2 == -1) {
  //       ix2 = sliceGroups[err];
  //       std::vector<int> tmpIx = groupStructure[ix2][0];
  //       groupStructure[ix2][0] = groupStructure[ix1][0];
  //       groupStructure[ix1][0] = tmpIx;
  //       break;
  //     }
  //   }
  // }
  // /*2) error: duplicate index in same group */
  // for (int err = 0; err < noOfSliceGroups; err++) {
  //   if (groupStructure[sliceGroups[err]].size() > 1) {
  //     groupStructure[sliceGroups[err]][0][0] = groupStructure[sliceGroups[err]][1][0];
  //     break;
  //   }
  // }
  // /*3) error: missing index in group */
  // for (int err = 0; err < noOfSliceGroups; err++) {
  //   if (groupStructure[sliceGroups[err]].size() > 1) {
  //     groupStructure[sliceGroups[err]].pop_back();
  //     break;
  //   }
  // }
  // /*4) error: duplicate index in other group */
  // int ix1 = -1;
  // int ix2 = -1;
  // for (int err = 0; err < noOfSliceGroups; err++) {
  //   if (groupStructure[sliceGroups[err]].size() > 1) {
  //     if (ix1 == -1)
  //       ix1 = sliceGroups[err];
  //     else if (ix2 == -1) {
  //       ix2 = sliceGroups[err];
  //       int tmpIx = groupStructure[ix2][0][0];
  //       groupStructure[ix2][0][0] = groupStructure[ix1][0][0];
  //       groupStructure[ix1][0][0] = tmpIx;
  //       break;
  //     }
  //   }
  // }

  // with these random groups, shuffle their rows and use for the slice:
  std::vector<int>ids;
  for (unsigned i = 0; i < sliceGroups.size(); i++) {
    int grpIndex = sliceGroups[i];
    std::shuffle(std::begin(groupStructure[grpIndex]), std::end(groupStructure[grpIndex]), _rndGen);
    for (unsigned row = 0; row < groupStructure[grpIndex].size(); row++) {
      ids.push_back(groupStructure[grpIndex][row][0]);
    }
  }

  DMatrixHandle dmSlice;
  XGDMatrixSliceDMatrix(dmAll, &ids[0], ids.size(), &dmSlice);

  // if slice is created without error (as expected), do a final check to ensure sliced information is the same 
  // as requested.  This can be done by simply comparing DMatrix column 0 with 'ids' array (as provided to slice method)
  xgboost::data::SimpleCSRSource src;
  src.CopyFrom(static_cast<std::shared_ptr<xgboost::DMatrix>*>(dmSlice)->get());
  bool failed = false;
  int ix = 0;
  for (unsigned i = 0; i < src.row_data_.size(); i += 2) {
    if (static_cast<int>(src.row_data_[i].fvalue != ids[ix])) {
      LOG(FATAL) << "Integrity test failed for XGIntegrityTests::GroupSlices(). Before and after comparisons failed!";
      // std::cout << "Error this should never occur!!! Place breakpoint and debug!";
      // failed = true;
      // break;
    }
    ix++;
  }

  // now, most importantly, check the groups match:
  if (!failed) {
    unsigned cumlSliceGroupAmt = 0;
    for (unsigned grpCumlIndex = 1; grpCumlIndex < src.info.group_ptr.size(); grpCumlIndex++) {
      cumlSliceGroupAmt += static_cast<unsigned>(groupStructure[sliceGroups[grpCumlIndex - 1]].size());
      if (cumlSliceGroupAmt != src.info.group_ptr[grpCumlIndex]) {
        LOG(FATAL) << "Integrity test failed for XGIntegrityTests::GroupSlices(). Group sizes do not match!";
        // std::cout << "Error with group sizes! This should never occur!!! Place breakpoint and debug!";
        // failed = true;
        // break;
      }
      // std::cout << cumlSliceGroupAmt << " " << src.info.group_ptr[grpCumlIndex] << std::endl;
    }
  }
}
}  // namespace xgboost
