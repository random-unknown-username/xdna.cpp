#include "xdna/xrt_probe.h"

namespace xdna {

XrtProbe probe_xrt() {
    XrtProbe result;
    result.error = "xdna.cpp was built without XRT support";
    return result;
}

}  // namespace xdna

