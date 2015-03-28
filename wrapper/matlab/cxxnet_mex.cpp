
#include <cstdio>
#include <string>
#include <cstring>
#include <assert.h>
#include "mex.h"
#define CXXNET_IN_MATLAB
#include "cxxnet_wrapper.h"


typedef unsigned long long uint64;
union Ptr {
  uint64 data;
  void *ptr;
};


static mxArray* SetHandle(void *handle) {
  union Ptr bridge;
  bridge.data = 0;
  bridge.ptr = handle;
  const mwSize dims[1] = {1};
  mxArray *mx_out = mxCreateNumericArray(1, dims, mxUINT64_CLASS, mxREAL);
  uint64 *up = reinterpret_cast<uint64*>(mxGetData(mx_out));
  *up = bridge.data;
  return mx_out;
}

static void *GetHandle(const mxArray *input) {
  union Ptr bridge;
  uint64 *up = reinterpret_cast<uint64*>(mxGetData(input));
  bridge.data = *up;
  return bridge.ptr;
}

inline void Transpose(float *mat, cxx_uint w, cxx_uint h) {
  // Use https://en.wikipedia.org/wiki/In-place_matrix_transposition
  float tmp;
  for (cxx_uint start = 0; start < w * h; ++start) {
    cxx_uint next = start;
    cxx_uint i = 0;
    do {
      i++;
      next = (next % h) * w + next / h;
    } while (next > start);
    if (next < start || i == 1) continue;
    tmp = mat[next = start];
    do {
      i = (next % h) * w + next / h;
      mat[next] = (i == start) ? tmp : mat[i];
      next = i;
    } while (next > start);
  }
}

inline mxArray* Ctype2Mx4DT(const cxx_real_t *ptr, cxx_uint oshape[4], cxx_uint ostride) {
  // COL MAJOR PROBLEM
  const mwSize dims[4] = {oshape[0], oshape[1], oshape[2], oshape[3]};
  const cxx_uint cxx_stride = oshape[1] * oshape[2] * ostride;
  const cxx_uint mx_stride = oshape[1] * oshape[2] * oshape[3];
  mxArray *mx_out = mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
  cxx_real_t *mx_ptr = reinterpret_cast<cxx_real_t*>(mxGetData(mx_out));
  for (cxx_uint i = 0; i < oshape[0]; ++i) {
    cxx_real_t *inst_mx_ptr = mx_ptr + i * mx_stride;
    cxx_real_t *inst_cxx_ptr = const_cast<cxx_real_t*>(ptr) + i * cxx_stride;
    for (cxx_uint j = 0; j < oshape[1]; ++j) {
      cxx_real_t *mat_mx_ptr = inst_mx_ptr + j * oshape[2] * oshape[3];
      cxx_real_t *mat_cxx_ptr = inst_cxx_ptr + j * oshape[2] * ostride;
      for (cxx_uint m = 0; m < oshape[2]; ++m) {
        for (cxx_uint n = 0; n < oshape[3]; ++n) {
          mat_mx_ptr[n * oshape[2] + m] = mat_cxx_ptr[m * ostride + n];
        }
      }
    }
  }
  return mx_out;
}

inline mxArray* Ctype2Mx2DT(const cxx_real_t *ptr, cxx_uint oshape[2], cxx_uint ostride) {
  const mwSize dims[2] = {oshape[0], oshape[1]};
  mxArray *mx_out = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxREAL);
  cxx_real_t *mx_ptr = reinterpret_cast<cxx_real_t*>(mxGetPr(mx_out));
  for (cxx_uint i = 0; i < oshape[0]; ++i) {
    for (cxx_uint j = 0; j < oshape[1]; ++j) {
      mx_ptr[j * oshape[0] + i] = ptr[i * ostride + j];
    }
  }
  return mx_out;
}


inline mxArray* Ctype2Mx1DT(const cxx_real_t *ptr, cxx_uint len) {
  const mwSize dims[1] = {len};
  mxArray *mx_out = mxCreateNumericArray(1, dims, mxSINGLE_CLASS, mxREAL);
  cxx_real_t *mx_ptr = reinterpret_cast<cxx_real_t*>(mxGetData(mx_out));
  memcpy(mx_ptr, ptr, len * sizeof(cxx_real_t));
  return mx_out;
}



static void MEXCXNIOCreateFromConfig(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  char *conf = mxArrayToString(prhs[1]);
  void *handle = CXNIOCreateFromConfig(conf);
  plhs[0] = SetHandle(handle);
  mxFree(conf);
}

static void MEXCXNIONext(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  const mwSize dims[1] = {1};
  mxArray *mx_out = mxCreateNumericArray(1, dims, mxINT32_CLASS, mxREAL);
  int *mx_ptr = reinterpret_cast<int*>(mxGetData(mx_out));
  int res = CXNIONext(handle);
  memcpy(mx_ptr, &res, sizeof(int));
  plhs[0] = mx_out;
}

static void MEXCXNIOBeforeFirst(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  CXNIOBeforeFirst(handle);
}

static void MEXCXNIOGetData(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  cxx_uint oshape[4];
  cxx_uint ostride = 0;
  const cxx_real_t *res_ptr = CXNIOGetData(handle, oshape, &ostride);
  plhs[0] = Ctype2Mx4DT(res_ptr, oshape, ostride);
}

static void MEXCXNIOGetLabel(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  cxx_uint oshape[2];
  cxx_uint ostride = 0;
  const cxx_real_t *res_ptr = CXNIOGetLabel(handle, oshape, &ostride);
  plhs[0] = Ctype2Mx2DT(res_ptr, oshape, ostride);
}

static void MEXCXNIOFree(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  CXNIOFree(handle);
}

static void MEXCXNNetCreate(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  char *dev = mxArrayToString(prhs[1]);
  char *conf = mxArrayToString(prhs[2]);
  void *handle = CXNNetCreate(dev, conf);
  plhs[0] = SetHandle(handle);
  mxFree(dev);
  mxFree(conf);
}

static void MEXCXNNetFree(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  CXNNetFree(handle);
}

static void MEXCXNNetSetParam(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  char *key = mxArrayToString(prhs[2]);
  char *val = mxArrayToString(prhs[3]);
  CXNNetSetParam(handle, key, val);
  mxFree(key);
  mxFree(val);
}

static void MEXCXNNetInitModel(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  CXNNetInitModel(handle);
}

static void MEXCXNNetSaveModel(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  char *fname = mxArrayToString(prhs[2]);
  CXNNetSaveModel(handle, fname);
  mxFree(fname);
}

static void MEXCXNNetLoadModel(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  char *fname = mxArrayToString(prhs[2]);
  CXNNetLoadModel(handle, fname);
  mxFree(fname);
}

static void MEXCXNNetStartRound(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  int *ptr = reinterpret_cast<int*>(mxGetData(prhs[2]));
  CXNNetStartRound(handle, *ptr);
}

static void MEXCXNNetSetWeight(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  const mxArray *p_weight = prhs[2];
  char *layer_name = mxArrayToString(prhs[3]);
  char *wtag = mxArrayToString(prhs[4]);
  cxx_uint size = 1;
  cxx_real_t *ptr = reinterpret_cast<cxx_real_t*>(mxGetData(p_weight));
  cxx_uint dims = mxGetNumberOfDimensions(p_weight);
  const mwSize *shape = mxGetDimensions(p_weight);
  for (cxx_uint i = 0; i < dims; ++i) {
    size *= shape[i];
  }
  if (dims == 2) {
    Transpose(ptr, shape[0], shape[1]);
  } else if (dims == 4) {
    for (int i = 0; i < shape[0]; ++i) {
      for (int j = 0;j < shape[1]; ++j) {
        cxx_real_t *mat = ptr + i * shape[1] * shape[2] * shape[3] + j * shape[2] * shape[3];
        Transpose(mat, shape[2], shape[3]);
      }
    }
  }
  CXNNetSetWeight(handle, ptr, size, layer_name, wtag);
  mxFree(layer_name);
  mxFree(wtag);
}

static void MEXCXNNetGetWeight(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  char *layer_name = mxArrayToString(prhs[2]);
  char *wtag = mxArrayToString(prhs[3]);
  cxx_uint wshape[4] = {0};
  cxx_uint odim = 0;
  const cxx_real_t *res_ptr = CXNNetGetWeight(handle, layer_name, wtag, wshape, &odim);
  if (odim == 0) res_ptr = NULL;
  if (wshape[3] != 0) {
    plhs[0] = Ctype2Mx4DT(res_ptr, wshape, wshape[3]);
  } else if (wshape[1] != 0) {
    cxx_uint shape[2] = {wshape[0], wshape[1]};
    plhs[0] = Ctype2Mx2DT(res_ptr, shape, shape[1]);
  } else {
    plhs[0] = Ctype2Mx1DT(res_ptr, wshape[0]);
  }
  mxFree(layer_name);
  mxFree(wtag);
}

static void MEXCXNNetUpdateIter(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  void *data_handle = GetHandle(prhs[2]);
  CXNNetUpdateIter(handle, data_handle);
}

static void MEXCXNNetUpdateBatch(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  // COL MAJOR PROBLEM
  void *handle = GetHandle(prhs[1]);
  const mxArray *p_data = prhs[2];
  const mxArray *p_label = prhs[3];
  cxx_uint dshape[4];
  cxx_uint lshape[2];
  assert(mxGetNumberOfDimensions(p_data) == 4);
  assert(mxGetNumberOfDimensions(p_label) == 2);
  const mwSize *d_size = mxGetDimensions(p_data);
  const mwSize *l_size = mxGetDimensions(p_label);
  for (int i = 0; i < 4; ++i) dshape[i] = d_size[i];
  for (int i = 0; i < 2; ++i) lshape[i] = l_size[i];
  cxx_real_t *ptr_data = reinterpret_cast<cxx_real_t*>(mxGetData(p_data));
  cxx_real_t *ptr_label = reinterpret_cast<cxx_real_t*>(mxGetData(p_label));
  for (cxx_uint i = 0; i < dshape[0]; ++i) {
    for (cxx_uint j = 0; j < dshape[1]; ++j) {
      float *mat = ptr_data + i * (dshape[1] * dshape[2] * dshape[3]) + j * (dshape[2] * dshape[3]);
      Transpose(mat, dshape[2], dshape[3]);
    }
  }
  Transpose(ptr_label, lshape[0], lshape[1]);
  CXNNetUpdateBatch(handle, ptr_data, dshape, ptr_label, lshape);
}

static void MEXCXNNetPredictBatch(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  const mxArray *p_data = prhs[2];
  cxx_uint dshape[4];
  assert(mxGetNumberOfDimensions(p_data) == 4);
  const mwSize *d_size = mxGetDimensions(p_data);
  for (int i = 0; i < 4; ++i) dshape[i] = d_size[i];
  cxx_uint out_size = 0;
  cxx_real_t *ptr_data = reinterpret_cast<cxx_real_t*>(mxGetData(p_data));
  const cxx_real_t *ptr_res = CXNNetPredictBatch(handle, ptr_data, dshape, &out_size);
  plhs[0] = Ctype2Mx1DT(ptr_res, out_size);
}

static void MEXCXNNetPredictIter(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  void *data_handle = GetHandle(prhs[2]);
  cxx_uint out_size = 0;
  const cxx_real_t *ptr_res = CXNNetPredictIter(handle, data_handle, &out_size);
  plhs[0] = Ctype2Mx1DT(ptr_res, out_size);
}

static void MEXCXNNetExtractBatch(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  const mxArray *p_data = prhs[2];
  char *node_name = mxArrayToString(prhs[3]);
  assert(mxGetNumberOfDimensions(p_data) == 4);
  cxx_uint dshape[4];
  const mwSize *d_size = mxGetDimensions(p_data);
  for (int i = 0; i < 4; ++i) dshape[i] = d_size[i];
  cxx_uint oshape[4];
  cxx_real_t *ptr_data = reinterpret_cast<cxx_real_t*>(mxGetData(p_data));
  const cxx_real_t *ptr_res = CXNNetExtractBatch(handle, ptr_data, dshape, node_name, oshape);
  plhs[0] = Ctype2Mx4DT(ptr_res, oshape, oshape[3]);
  mxFree(node_name);
}

static void MEXCXNNetExtractIter(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  void *data_handle = GetHandle(prhs[2]);
  char *node_name = mxArrayToString(prhs[3]);
  cxx_uint oshape[4];
  const cxx_real_t *ptr_res = CXNNetExtractIter(handle, data_handle, node_name, oshape);
  plhs[0] = Ctype2Mx4DT(ptr_res, oshape, oshape[3]);
  mxFree(node_name);
}

static void MEXCXNNetEvaluate(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  void *data_handle = GetHandle(prhs[2]);
  char *data_name = mxArrayToString(prhs[3]);
  const char *ret = CXNNetEvaluate(handle, data_handle, data_name);
  printf("%s\n", ret);
  plhs[0] = mxCreateString(ret);
  mxFree(data_name);
}


// MEX Function
//

struct handle_registry {
  std::string cmd;
  void (*func)(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs);
};


static handle_registry handles[] = {
  {"MEXCXNIOCreateFromConfig", MEXCXNIOCreateFromConfig},
  {"MEXCXNIONext", MEXCXNIONext},
  {"MEXCXNIOBeforeFirst", MEXCXNIOBeforeFirst},
  {"MEXCXNIOGetData", MEXCXNIOGetData},
  {"MEXCXNIOGetLabel", MEXCXNIOGetLabel},
  {"MEXCXNIOFree", MEXCXNIOFree},
  {"MEXCXNNetCreate", MEXCXNNetCreate},
  {"MEXCXNNetFree", MEXCXNNetFree},
  {"MEXCXNNetSetParam", MEXCXNNetSetParam},
  {"MEXCXNNetInitModel", MEXCXNNetInitModel},
  {"MEXCXNNetSaveModel", MEXCXNNetSaveModel},
  {"MEXCXNNetLoadModel", MEXCXNNetLoadModel},
  {"MEXCXNNetStartRound", MEXCXNNetStartRound},
  {"MEXCXNNetSetWeight", MEXCXNNetSetWeight},
  {"MEXCXNNetGetWeight", MEXCXNNetGetWeight},
  {"MEXCXNNetUpdateIter", MEXCXNNetUpdateIter},
  {"MEXCXNNetUpdateBatch", MEXCXNNetUpdateBatch},
  {"MEXCXNNetPredictBatch", MEXCXNNetPredictBatch},
  {"MEXCXNNetPredictIter", MEXCXNNetPredictIter},
  {"MEXCXNNetExtractBatch", MEXCXNNetExtractBatch},
  {"MEXCXNNetExtractIter", MEXCXNNetExtractIter},
  {"MEXCXNNetEvaluate", MEXCXNNetEvaluate},
  {"NULL", NULL},
};

void mexFunction(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  mexLock();  // Avoid clearing the mex file.
  if (nrhs == 0) {
    mexErrMsgTxt("No API command given");
    return;
  }
  char *cmd = mxArrayToString(prhs[0]);
  bool dispatched = false;
  for (int i = 0; handles[i].func != NULL; i++) {
    if (handles[i].cmd.compare(cmd) == 0) {
      handles[i].func(nlhs, plhs, nrhs, prhs);
      dispatched = true;
      break;
    }
  }
  if (!dispatched) {
    std::string err = "Unknown command '";
    err += cmd;
    err += "'";
    mexErrMsgTxt(err.c_str());
  }
  mxFree(cmd);
}
