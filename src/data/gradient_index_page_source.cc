#include "gradient_index_page_source.h"

namespace xgboost {
namespace data {
void GradientIndexPageSource::Fetch() {
  if (!this->ReadCache()) {
    auto const& csr = source_->Page();
    this->page_.reset(new GHistIndexMatrix());
    this->page_->Init(*csr, cuts_, is_dense_, nthreads_);
    this->WriteCache();
  }
}
}  // namespace data
}  // namespace xgboost
