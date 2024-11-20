// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tensor_external_data_info.h"
#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/common/string_utils.h"
#include "core/platform/path_lib.h"

#include <vector>

#ifdef _WIN32
#include <Windows.h>
#endif
using ::google::protobuf::RepeatedPtrField;
using ::ONNX_NAMESPACE::StringStringEntryProto;

namespace onnxruntime {
Status ExternalDataInfo::Create(const RepeatedPtrField<StringStringEntryProto>& input,
                                std::unique_ptr<ExternalDataInfo>& out) {
  auto str_to_int = [](const std::string& s, OFFSET_TYPE& result) -> Status {
    char* end;
#ifdef _WIN32
    result = _strtoi64(s.c_str(), &end, 10);
#else
    result = OrtStrToPtrDiff(s.c_str(), &end);
#endif
    if (end != s.c_str() + s.length()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "parsing ", s, " failed");
    }
    return Status::OK();
  };

  out = std::make_unique<ExternalDataInfo>();
  const int input_size = input.size();

  for (int i = 0; i != input_size; ++i) {
    StringStringEntryProto stringmap = input[i];
    if (!stringmap.has_key())
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "model format error! Need a key for the external data info");
    if (!stringmap.has_value())
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "model format error! Need a value for the external data info");
    if (stringmap.key() == "location" && !stringmap.value().empty()) {
      out->rel_path_ = ToWideString(stringmap.value());
    } else if (stringmap.key() == "offset" && !stringmap.value().empty()) {
      ORT_RETURN_IF_ERROR(str_to_int(stringmap.value(), out->offset_));
    } else if (stringmap.key() == "length" && !stringmap.value().empty()) {
      char* end;
      out->length_ = narrow<size_t>(OrtStrToPtrDiff(stringmap.value().c_str(), &end));
      if (end != stringmap.value().c_str() + stringmap.value().length())
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "parsing ", stringmap.value(), " failed");
    } else if (stringmap.key() == "checksum" && !stringmap.value().empty()) {
      out->checksum_ = stringmap.value();
    } else if (stringmap.key().find("prepacked", 0) == 0) {
      // Starts with 'prepacked'. Each prepacked entry may have multiple blobs with the same key
      //  we output them with the same key
      // format = key|offset;length;checksum[|offset;length;checksum]
      // We are ignoring invalid entries (should not be any), and rely
      // on in memory pre-packs regenerated in this case.
      // users can over-write this file with the correct pre-packed info.
      const std::string& prepacked = stringmap.value();
      if (!prepacked.empty()) {
        PrepackedInfos prepacked_infos;
        auto split_fields = utils::SplitString(prepacked, "|", false);
        if (split_fields.size() > 2) {
          const std::string key{split_fields[0]};
          auto& blob_infos = prepacked_infos[key];
          for (size_t f = 1; f < split_fields.size(); ++f) {
            const auto& blob = split_fields[f];
            auto blob_fields = utils::SplitString(blob, ";", false);
            if (blob_fields.size() == 3) {
              OFFSET_TYPE offset, len;
              ORT_RETURN_IF_ERROR(str_to_int(std::string(blob_fields[0]), offset));
              ORT_RETURN_IF_ERROR(str_to_int(std::string(blob_fields[1]), len));
              blob_infos.push_back(std::make_tuple(offset, narrow<size_t>(len), std::string(blob_fields[2])));
            }
          }
          if (blob_infos.empty()) {
            prepacked_infos.erase(key);
          }
        }
        if (!prepacked_infos.empty()) {
          out->prepacked_infos_ = std::move(prepacked_infos);
        }
      }
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "model format error!");
    }
  }
  if (out->rel_path_.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "model format error! Missing 'location'");
  }
  return Status::OK();
}
}  // namespace onnxruntime