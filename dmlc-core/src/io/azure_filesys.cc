/*!
 *  Copyright (c) 2015 by Contributors
 * \file azure_filesys.cc
 * \brief Azure access module
 * \author Mu Li
 */
#include "./azure_filesys.h"
#include "stdafx.h"

#include "was/storage_account.h"
#include "was/blob.h"
#include "cpprest/filestream.h"
#include "cpprest/containerstream.h"

namespace dmlc {
namespace io {

namespace {
std::vector<std::string> split(std::string str, char delimiter) {
  std::vector<std::string> internal;
  std::stringstream ss(str);
  std::string tok;

  while (std::getline(ss, tok, delimiter)) {
    internal.push_back(tok);
  }
  return internal;
}
}  // namespace

AzureFileSystem::AzureFileSystem() {
  const char *name = getenv("AZURE_STORAGE_ACCOUNT");
  const char* key = getenv("AZURE_STORAGE_ACCESS_KEY");
  CHECK_NE(name, NULL)
      << "Need to set enviroment variable AZURE_STORAGE_ACCOUNT to use Azure";
  CHECK_NE(key, NULL)
      << "Need to set enviroment variable AZURE_STORAGE_ACCESS_KEY to use Azure";
  azure_account_ = name;
  azure_key_ = key;
}

void AzureFileSystem::ListDirectory(
    const URI &path, std::vector<FileInfo> *out_list) {
  CHECK(path.host.length()) << "container name not specified in azure";
  out_list->clear();

  utility::string_t
      storage_connection_string(U("DefaultEndpointsProtocol=https;AccountName="
      + azure_account_ + ";AccountKey= " + azure_key_));

  // Retrieve storage account from connection string.
  azure::storage::cloud_storage_account storage_account
      = azure::storage::cloud_storage_account::parse(storage_connection_string);

  // Create the blob client.
  azure::storage::cloud_blob_client blob_client
      = storage_account.create_cloud_blob_client();

  // Retrieve a reference to a previously created container.
  azure::storage::cloud_blob_container container
      = blob_client.get_container_reference(U("container"));

  // Output URI of each item.
  azure::storage::list_blob_item_iterator end_of_results;
  for (auto it = container.list_blobs(); it != end_of_results; ++it) {
    if (it->is_blob()) {
      ucout << U("Blob: ") << it->as_blob().uri().primary_uri().to_string() << std::endl;
      FileInfo info;
      info.path = path;
      size_t value = it->as_blob().properties().size();
      info.size = static_cast<size_t>(value);
      std::vector<std::string> splitVec
          = split(it->as_blob().uri().primary_uri().to_string(), '/');
      info.path.name = '/' + splitVec[splitVec.size()-1];
      info.type = kFile;
      out_list->push_back(info);
    } else {
      ucout << U("Directory: ") << it->as_directory().uri().primary_uri().to_string() << std::endl;
      FileInfo info;
      info.path = path;
      info.size = 0;
      std::vector<std::string> splitVec =
          split(it->as_directory().uri().primary_uri().to_string(), '/');
      info.path.name = '/' + splitVec[splitVec.size()-1];
      info.type = kDirectory;
      out_list->push_back(info);
    }
  }
}

}  // namespace io
}  // namespace dmlc
