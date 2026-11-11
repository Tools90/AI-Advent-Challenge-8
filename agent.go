package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"
)

type TraceStep struct {
	Type             string `json:"type"`
	Tool             string `json:"tool,omitempty"`
	Arguments        string `json:"arguments,omitempty"`
	Result           string `json:"result,omitempty"`
	Request          string `json:"request,omitempty"`
	Response         string `json:"response,omitempty"`
	PromptTokens     int    `json:"promptTokens,omitempty"`
	CompletionTokens int    `json:"completionTokens,omitempty"`
	Error            string `json:"error,omitempty"`
	LatencyMs        int64  `json:"latencyMs"`
}

type RunResult struct {
	Reply                string
	Messages             []LLMMessage
	Steps                []TraceStep
	Usage                Usage
	RequestTokens        int
	LastPromptTokens     int
	LastCompletionTokens int
	MaxPromptTokens      int
	LatencyMs            int64
	Calls                int
}

const agentSystemPrompt = "Ты — агент с инструментами. Если для ответа нужны время, вычисления, конвертация валюты или данные о документах, вызови подходящий инструмент. " +
	"Для анализа содержания, темы, героев, сюжета или стиля книги обязательно читай её текст через read_document; инструмент возвращает весь текст сразу. " +
	"Не отвечай на вопросы о содержании книги только по общим знаниям: если текст не добавлен в контекст и не прочитан через read_document, сначала вызови инструмент. " +
	"Для точных подсчётов используй document_stats и search_document, затем при необходимости calculate. Не выдумывай числа, цитаты и факты. " +
	"Если документ уже добавлен в контекст, можешь опираться на его текст. После получения результатов дай финальный ответ пользователю по-русски."

type Agent struct {
	llm      LLMClient
	tools    map[string]Tool
	maxSteps int
}

func NewAgent(llm LLMClient, tools map[string]Tool, maxSteps int) *Agent {
	return &Agent{llm: llm, tools: tools, maxSteps: maxSteps}
}

func (a *Agent) Tools() []ToolSchema {
	names := make([]string, 0, len(a.tools))
	for name := range a.tools {
		names = append(names, name)
	}
	sort.Strings(names)

	out := make([]ToolSchema, 0, len(names))
	for _, name := range names {
		out = append(out, a.tools[name].Schema)
	}
	return out
}

func (a *Agent) Run(ctx context.Context, history []LLMMessage, userMessage string) (RunResult, error) {
	start := time.Now()

	system := agentSystemPrompt

	messages := make([]LLMMessage, 0, len(history)+2)
	messages = append(messages, LLMMessage{Role: "system", Content: system})
	messages = append(messages, history...)
	messages = append(messages, LLMMessage{Role: "user", Content: userMessage})

	turnStart := 1 + len(history)
	newMessages := func() []LLMMessage {
		return append([]LLMMessage(nil), messages[turnStart:]...)
	}

	schemas := a.Tools()
	var result RunResult

	for step := 0; step < a.maxSteps; step++ {
		callStart := time.Now()
		requestTrace := a.llm.TraceRequest(messages, schemas)
		response, err := a.llm.Chat(ctx, messages, schemas)
		llmTrace := TraceStep{
			Type:             "llm",
			Request:          requestTrace,
			PromptTokens:     response.Usage.PromptTokens,
			CompletionTokens: response.Usage.CompletionTokens,
			LatencyMs:        time.Since(callStart).Milliseconds(),
		}
		result.LastPromptTokens = response.Usage.PromptTokens
		result.LastCompletionTokens = response.Usage.CompletionTokens
		if response.Usage.PromptTokens > result.MaxPromptTokens {
			result.MaxPromptTokens = response.Usage.PromptTokens
		}

		if result.Calls == 0 {
			result.RequestTokens = response.Usage.PromptTokens
		}
		result.Calls++
		result.Usage.Add(response.Usage)
		if err != nil {
			llmTrace.Error = err.Error()
			result.Steps = append(result.Steps, llmTrace)
			result.LatencyMs = time.Since(start).Milliseconds()
			result.Messages = newMessages()
			return result, err
		}

		llmTrace.Response = traceResponse(response)
		result.Steps = append(result.Steps, llmTrace)
		messages = append(messages, response.Message)

		if len(response.Message.ToolCalls) == 0 {
			result.Reply = strings.TrimSpace(response.Message.Content)
			result.LatencyMs = time.Since(start).Milliseconds()
			result.Messages = newMessages()
			if result.Reply == "" {
				return result, errors.New("модель вернула пустой ответ")
			}
			return result, nil
		}

		for _, call := range response.Message.ToolCalls {
			stepStart := time.Now()
			trace := TraceStep{
				Type:      "tool",
				Tool:      call.Function.Name,
				Arguments: call.Function.Arguments,
				LatencyMs: time.Since(stepStart).Milliseconds(),
			}

			content := ""
			tool, ok := a.tools[call.Function.Name]
			if !ok {
				trace.Error = fmt.Sprintf("неизвестный инструмент %q", call.Function.Name)
				content = toolErrorContent(trace.Error)
			} else {
				value, err := tool.Run(ctx, json.RawMessage(call.Function.Arguments))
				if err != nil {
					trace.Error = err.Error()
					content = toolErrorContent(err.Error())
				} else {
					encoded, err := json.Marshal(value)
					if err != nil {
						trace.Error = err.Error()
						content = toolErrorContent(err.Error())
					} else {
						trace.Result = string(encoded)
						content = toolOKContent(encoded)
					}
				}
			}

			trace.LatencyMs = time.Since(stepStart).Milliseconds()
			result.Steps = append(result.Steps, trace)
			messages = append(messages, LLMMessage{
				Role:       "tool",
				ToolCallID: call.ID,
				Name:       call.Function.Name,
				Content:    content,
			})
		}
	}

	result.LatencyMs = time.Since(start).Milliseconds()
	result.Messages = newMessages()
	return result, fmt.Errorf("агент превысил лимит шагов (%d)", a.maxSteps)
}

func toolOKContent(result json.RawMessage) string {
	return `{"ok":true,"result":` + string(result) + `}`
}

func toolErrorContent(message string) string {
	encoded, _ := json.Marshal(map[string]any{"ok": false, "error": message})
	return string(encoded)
}

func traceResponse(response LLMResponse) string {
	payload := map[string]any{
		"message":       response.Message,
		"finish_reason": response.FinishReason,
		"usage":         response.Usage,
	}
	encoded, err := json.Marshal(payload)
	if err != nil {
		return err.Error()
	}
	return string(encoded)
}
