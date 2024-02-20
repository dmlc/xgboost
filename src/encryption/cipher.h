#pragma once

#include <map>
#include <any>
#include <xgboost/span.h>

namespace xgboost::encryption {

    const std::string NEED_ENCRYPTION = "need_encryption"; // An integer

    /*! \brief An encryption cipher interface for distributed XGBoost */
    class Cipher {

        virtual void Initialize(std::map<std::string, std::string> params) = 0;

        virtual void Shutdown() = 0;

        virtual std::any Encrypt(std::vector<double> cleartext) = 0;

        virtual std::vector<double> Decrypt(std::any ciphertext) = 0;

        virtual std::any Add(std::any cipher_vec1, std::any cipher_vec2) = 0;

        virtual std::any AddClear(std::any cipher_vec, std::vector<double> clear_vec) = 0;

        virtual std::any Multiply(std::any cipher_vec1, std::any cipher_vec2) = 0;

        virtual std::any MultiplyClear(std::any cipher_vec, std::vector<double> clear_vec) = 0;

        virtual common::Span<std::int8_t> SerializeVector(
                std::vector<std::any> vec, int data_size, std::map<std::string, std::string> headers) = 0;

        virtual std::vector<std::any> DeserializeVector(
                common::Span<std::int8_t> buffer, std::map<std::string, std::string> &headers) = 0;

    };

    class CipherLoader {

    private:
        std::map<std::string, std::string> params;


    public:
        CipherLoader(): params(NULL) {}

        CipherLoader(std::map<std::string, std::string>& params): params(params) {}

        Cipher load(std::string plugin_name);

    };

}  // namespace xgboost::encryption
