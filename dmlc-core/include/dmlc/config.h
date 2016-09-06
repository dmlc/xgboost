/*!
 * Copyright (c) 2015 by Contributors
 * \file config.h
 * \brief defines config parser class
 */
#ifndef DMLC_CONFIG_H_
#define DMLC_CONFIG_H_

#include <cstring>
#include <iostream>
#include <iterator>
#include <map>
#include <vector>
#include <utility>
#include <string>
#include <sstream>

/*! \brief namespace for dmlc */
namespace dmlc {

/*!
 * \brief class for config parser
 *
 * Two modes are supported:
 * 1. non-multi value mode: if two same keys in the configure file, the later one will replace the
 *      ealier one; when using iterator, the order will be the "last effective insersion" order
 * 2. multi value mode: multiple values with the same key could co-exist; when using iterator, the
 *      order will be the insersion order.
 *
 * [Basic usage]
 *
 * Config cfg(file_input_stream);
 * for(Config::ConfigIterator iter = cfg.begin(); iter != cfg.end(); ++iter) {
 *     ConfigEntry ent = *iter;
 *     std::string key = ent.first;
 *     std::string value = ent.second;
 *     do_something_with(key, value);
 * }
 */
class Config {
 public:
  /*!
   * \brief type when extracting from iterator
   */
  typedef std::pair<std::string, std::string> ConfigEntry;

  /*!
   * \brief iterator class
   */
  class ConfigIterator;

  /*!
   * \brief create empty config
   * \param multi_value whether the config supports multi value
   */
  explicit Config(bool multi_value = false);
  /*!
   * \brief create config and load content from the given stream
   * \param is input stream
   * \param multi_value whether the config supports multi value
   */
  explicit Config(std::istream& is, bool multi_value = false);  // NOLINT(*)
  /*!
   * \brief clear all the values
   */
  void Clear(void);
  /*!
   * \brief load the contents from the stream
   * \param is the stream as input
   */
  void LoadFromStream(std::istream& is);  // NOLINT(*)
  /*!
   * \brief set a key-value pair into the config; if the key already exists in the configure file,
   *        it will either replace the old value with the given one (in non-multi value mode) or
   *        store it directly (in multi-value mode);
   * \param key key
   * \param value value
   * \param is_string whether the value should be wrapped by quotes in proto string
   */
  template<class T>
  void SetParam(const std::string& key, const T& value, bool is_string = false);

  /*!
   * \brief get the config under the key; if multiple values exist for the same key,
   *        return the last inserted one.
   * \param key key
   * \return config value
   */
  const std::string& GetParam(const std::string& key) const;

  /*!
   * \brief check whether the configure value given by the key should be wrapped by quotes
   * \param key key
   * \return whether the configure value is represented by string
   */
  bool IsGenuineString(const std::string& key) const;

  /*!
   * \brief transform all the configuration into string recognizable to protobuf
   * \return string that could be parsed directly by protobuf
   */
  std::string ToProtoString(void) const;

  /*!
   * \brief get begin iterator
   * \return begin iterator
   */
  ConfigIterator begin() const;

  /*!
   * \brief get end iterator
   * \return end iterator
   */
  ConfigIterator end() const;

 public:
  /*!
   * \brief iterator class
   */
  class ConfigIterator : public std::iterator< std::input_iterator_tag, ConfigEntry > {
    friend class Config;
   public:
    /*!
     * \brief copy constructor
     */
    ConfigIterator(const ConfigIterator& other);
    /*!
     * \brief uni-increment operators
     * \return the reference of current config
     */
    ConfigIterator& operator++();
    /*!
     * \brief uni-increment operators
     * \return the reference of current config
     */
    ConfigIterator operator++(int);  // NOLINT(*)
    /*!
     * \brief compare operators
     * \param rhs the other config to compare against
     * \return the compared result
     */
    bool operator == (const ConfigIterator& rhs) const;
    /*!
     * \brief compare operators not equal
     * \param rhs the other config to compare against
     * \return the compared result
     */
    bool operator != (const ConfigIterator& rhs) const;
    /*!
     * \brief retrieve value from operator
     */
    ConfigEntry operator * () const;

   private:
    ConfigIterator(size_t index, const Config* config);
    void FindNextIndex();

   private:
    size_t index_;
    const Config* config_;
  };

 private:
  struct ConfigValue {
    std::vector<std::string> val;
    std::vector<size_t> insert_index;
    bool is_string;
  };
  void Insert(const std::string& key, const std::string& value, bool is_string);

 private:
  std::map<std::string, ConfigValue> config_map_;
  std::vector<std::pair<std::string, size_t> > order_;
  const bool multi_value_;
};

template<class T>
void Config::SetParam(const std::string& key, const T& value, bool is_string) {
  std::ostringstream oss;
  oss << value;
  Insert(key, oss.str(), is_string);
}

}  // namespace dmlc

#endif  // DMLC_CONFIG_H_
