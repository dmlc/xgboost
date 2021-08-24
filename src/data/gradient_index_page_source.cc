#include "gradient_index_page_source.h"

namespace xgboost {
namespace data {
void GradientIndexPageSource::Fetch() {
  if (!this->ReadCache()) {
    auto const& csr = source_->Page();
    this->page_.reset(new GHistIndexMatrix());
    this->page_->Init(*csr, cuts_);
    this->page_->base_rowid = csr->base_rowid;
    this->WriteCache();
  }
}
}  // namespace data
}  // namespace xgboost
