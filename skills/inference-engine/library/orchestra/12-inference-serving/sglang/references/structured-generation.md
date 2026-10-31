# SGLang 结构生成：API、语法与完成状态

核对的`sg.gen`签名提供regex与json_schema，不接受任意tools或grammar参数。服务HTTP层支持的grammar功能可能另有请求字段/后端；不能将其直接当作前端gen关键字，也不能把Lark对象直接传给不存在的grammar参数。

## JSON与schema
使用支持的schema接口和实际backend，例如包含type/properties/required的JSON schema；约束keyword支持范围需确认。简单`\{[^}]+\}`既不是JSON解析器，也不能正确处理嵌套、引号与转义。生成后按完整schema及业务要求验证。

schema中的format、日期正则、邮箱模式只能保证各自支持的格式范围，不能确认真实日期、邮箱存在或事实正确。缺失字段应允许null/失败或由业务处理，不编造值。

## grammar与工具
CFG/EBNF的terminal、tokenizer映射、编译和每请求parser状态需要真实backend接线。旧文中未定义缩进终结符的Python grammar及SQL示例不能宣称语法保证；生成的代码还需要解析/测试，SQL需约定语法与参数绑定。

工具调用走实际支持的tool protocol，执行映射由应用控制，不从字符串直接eval。文本提示生成SELECT/WHERE不构成完整SQL合法性证明。

## streaming、retry与计时
检查finish_reason、max_tokens、stop/cancel与服务错误；JSON可能截断。streaming partial只在最终完整校验后当成最终结果，chunk不等于JSON边界。

重试要执行真实解析与判定，有效后停止；旧示例只有“validate/break”注释实际会每次继续，不是完整retry实现。宽松fallback需显式条件，不能每次同时生成strict与loose再宣称只在失败触发。
记录首次/最终成功、重试请求数、tokens与端到端延迟。schema编译冷成本与复用后的成本分开；compiled grammar可共享，每个请求的可变状态需隔离。

程序批量运行使用真实program.run_batch。结合请求状态重排、speculative回滚和并发验证，不以一次JSON成功推导整个服务稳定。
