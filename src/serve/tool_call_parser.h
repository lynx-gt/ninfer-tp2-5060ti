#pragma once

#include "serve/request.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

// 模型输出了 tool 标记但结构不可表示时，Serve 按原文返回（协议不伪造 tool_call，也不报错）。
// 这个分类是机读契约：同时进 request_done JSON 与运维告警，供客户端/运维判断
// 「想调工具但语法坏了」与「正常终答」；只给稳定分类名，不带任何 markup 或参数内容。
enum class ToolCallFallbackReason : std::uint8_t {
    None,
    UnterminatedToolCall, // 见过 <tool_call> 但没有配对的 </tool_call>
    MalformedToolCall,    // 标记闭合了，但调用块内部结构/函数名不可表示
    TrailingContent,      // 完整调用之后还有无法归入调用块的尾部内容
};

[[nodiscard]] const char* tool_call_fallback_reason_name(ToolCallFallbackReason reason) noexcept;

struct ParsedToolCallOutput {
    bool is_tool_call_response = false;
    std::string content;
    std::vector<ToolCall> tool_calls;
    bool tool_marker_seen                  = false;
    ToolCallFallbackReason fallback_reason = ToolCallFallbackReason::None;
};

ParsedToolCallOutput parse_qwen_tool_call_output(const std::string& text,
                                                 std::size_t max_tool_name_length);

// Incrementally publishes text that is provably outside a possible Qwen
// <tool_call> suffix. At terminal time, a valid tool response discards the
// buffered tool region; malformed/non-tool output flushes it verbatim.
class ToolCallStreamFilter {
public:
    std::string feed(std::string_view text);
    std::string finish(bool is_tool_call_response);

    [[nodiscard]] std::size_t emitted_bytes() const noexcept { return emitted_bytes_; }

private:
    std::string pending_;
    std::string tool_region_;
    std::size_t emitted_bytes_ = 0;
    bool saw_tool_marker_      = false;
    bool finished_             = false;
};

} // namespace ninfer::serve
