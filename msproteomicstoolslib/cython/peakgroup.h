#ifndef MSPROTEOMICSTOOLSLIB_PEAKGROUP_H
#define MSPROTEOMICSTOOLSLIB_PEAKGROUP_H

#include <string>
#include <vector>
#include <stdexcept>

#include <iostream>

// forward decl
struct c_precursor;

struct c_peakgroup {

public:
  double fdr_score;
  double normalized_retentiontime;
  // ca 8 bytes per double
  // ca 24 bytes per empty string
  // ca 35 bytes extra for the full string (60 bytes total)
  std::string internal_id_; //  225284maxresident)k = 10 MB by itself (24 bytes)
  double intensity_;
  double dscore_;
  int cluster_id_;

  c_precursor* precursor;

  c_peakgroup() {};
  c_precursor* getPeptide() {return precursor;}

  // 245792maxresident)k   with
  // 218096maxresident)k w/o
  std::string getInternalId() {return internal_id_;}
  void setInternalId(std::string s) {internal_id_ = s;} // storing the string uses another 15 MB (35 bytes)

};

#endif
