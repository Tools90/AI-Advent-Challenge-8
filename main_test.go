package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

type mockLLM struct {
	responses []LLMResponse
	calls     int
	messages  [][]LLMMessage
}

func (m *mockLLM) Chat(ctx context.Context, messages []LLMMessage, tools []ToolSchema) (LLMResponse, error) {
	m.messages = append(m.messages, messages)
	index := m.calls
	m.calls++
	if index >= len(m.responses) {
		index = len(m.responses) - 1
	}
	return m.responses[index], nil
}

func (m *mockLLM) Model() string { return "mock-model" }

func (m *mockLLM) TraceRequest(messages []LLMMessage, tools []ToolSchema) string {
	data, _ := json.Marshal(map[string]any{"model": m.Model(), "tools": tools, "messages": messages})
	return string(data)
}

func toolCall(id, name, args string) ToolCall {
	call := ToolCall{ID: id, Type: "function"}
	call.Function.Name = name
	call.Function.Arguments = args
	return call
}

func TestCalculatorTool(t *testing.T) {
	cases := []struct {
		expr string
		want float64
		err  bool
	}{
		{"2+2", 4, false},
		{"17*23", 391, false},
		{"(2+3)*4", 20, false},
		{"10/4", 2.5, false},
		{"-5 + 2", -3, false},
		{"1/0", 0, true},
		{"2+", 0, true},
		{"abc", 0, true},
	}

	for _, c := range cases {
		args, _ := json.Marshal(map[string]string{"expression": c.expr})
		value, err := runCalcTool(context.Background(), args)
		if c.err {
			if err == nil {
				t.Errorf("%s: ожидалась ошибка", c.expr)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: %v", c.expr, err)
			continue
		}
		got := value.(CalcResult).Result
		if got != c.want {
			t.Errorf("%s = %v, ожидалось %v", c.expr, got, c.want)
		}
	}
}

func TestCurrencyTool(t *testing.T) {
	args := json.RawMessage(`{"amount":100,"from":"USD","to":"RUB"}`)
	value, err := runCurrencyTool(context.Background(), args)
	if err != nil {
		t.Fatal(err)
	}
	result := value.(CurrencyResult)
	if result.Result != 9250 {
		t.Errorf("result = %v", result.Result)
	}

	bad := json.RawMessage(`{"amount":1,"from":"GBP","to":"RUB"}`)
	if _, err := runCurrencyTool(context.Background(), bad); err == nil {
		t.Error("ожидалась ошибка для неизвестной валюты")
	}
}

func TestTimeTool(t *testing.T) {
	value, err := runTimeTool(context.Background(), json.RawMessage(`{"timezone":"Europe/Moscow"}`))
	if err != nil {
		t.Fatal(err)
	}
	if value.(TimeResult).Timezone != "Europe/Moscow" {
		t.Errorf("time = %+v", value)
	}
	if _, err := runTimeTool(context.Background(), json.RawMessage(`{"timezone":"Bad/Zone"}`)); err == nil {
		t.Error("ожидалась ошибка для неизвестной зоны")
	}
}

func TestAgentTwoTools(t *testing.T) {
	llm := &mockLLM{responses: []LLMResponse{
		{
			Message: LLMMessage{Role: "assistant", ToolCalls: []ToolCall{
				toolCall("c1", "calculate", `{"expression":"0.15*200"}`),
			}},
			Usage: Usage{PromptTokens: 7, CompletionTokens: 3, TotalTokens: 10},
		},
		{
			Message: LLMMessage{Role: "assistant", ToolCalls: []ToolCall{
				toolCall("c2", "convert_currency", `{"amount":30,"from":"USD","to":"RUB"}`),
			}},
			Usage: Usage{PromptTokens: 7, CompletionTokens: 3, TotalTokens: 10},
		},
		{
			Message: LLMMessage{Role: "assistant", Content: "30 долларов — это 2775 рублей."},
			Usage:   Usage{PromptTokens: 7, CompletionTokens: 3, TotalTokens: 10},
		},
	}}

	agent := NewAgent(llm, buildTools(nil), 6)
	result, err := agent.Run(context.Background(), nil, "Сколько рублей в 15% от 200 долларов?")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result.Reply != "30 долларов — это 2775 рублей." {
		t.Errorf("reply = %q", result.Reply)
	}
	if len(result.Steps) != 5 || result.Calls != 3 {
		t.Fatalf("steps=%d calls=%d", len(result.Steps), result.Calls)
	}
	if result.Steps[0].Type != "llm" || result.Steps[0].Request == "" || result.Steps[0].Response == "" {
		t.Errorf("llm trace = %+v", result.Steps[0])
	}
	if !strings.Contains(result.Steps[0].Request, "mock-model") || !strings.Contains(result.Steps[0].Request, "tools") {
		t.Errorf("request trace = %s", result.Steps[0].Request)
	}
	if result.Steps[1].Tool != "calculate" || result.Steps[3].Tool != "convert_currency" {
		t.Errorf("tool steps = %+v", result.Steps)
	}
	if !strings.Contains(result.Steps[1].Result, "30") || !strings.Contains(result.Steps[3].Result, "2775") {
		t.Errorf("steps = %+v", result.Steps)
	}
	if result.Usage.TotalTokens != 30 {
		t.Errorf("usage = %+v", result.Usage)
	}
	if result.LastPromptTokens != 7 || result.LastCompletionTokens != 3 {
		t.Errorf("last usage = prompt:%d completion:%d", result.LastPromptTokens, result.LastCompletionTokens)
	}
	if result.MaxPromptTokens != 7 {
		t.Errorf("max prompt tokens = %d", result.MaxPromptTokens)
	}
	if result.Steps[0].PromptTokens != 7 || result.Steps[0].CompletionTokens != 3 {
		t.Errorf("llm step usage = %+v", result.Steps[0])
	}
	if len(result.Messages) != 6 {
		t.Fatalf("turn messages = %d, ожидалось 6", len(result.Messages))
	}
	if result.Messages[0].Role != "user" || result.Messages[5].Role != "assistant" {
		t.Errorf("turn messages = %+v", result.Messages)
	}
	if len(result.Messages[1].ToolCalls) != 1 || result.Messages[1].ToolCalls[0].ID != "c1" {
		t.Errorf("assistant tool call = %+v", result.Messages[1])
	}
	if result.Messages[2].Role != "tool" || result.Messages[2].ToolCallID != "c1" || !strings.Contains(result.Messages[2].Content, "30") {
		t.Errorf("tool message = %+v", result.Messages[2])
	}

	// The second LLM call must contain the tool result for the first call.
	second := llm.messages[1]
	foundTool := false
	for _, message := range second {
		if message.Role == "tool" && message.ToolCallID == "c1" && strings.Contains(message.Content, "30") {
			foundTool = true
		}
	}
	if !foundTool {
		t.Error("tool result c1 не передан во второй вызов")
	}
}

func TestAgentMaxSteps(t *testing.T) {
	llm := &mockLLM{responses: []LLMResponse{
		{Message: LLMMessage{Role: "assistant", ToolCalls: []ToolCall{toolCall("c1", "get_current_time", `{}`)}}},
	}}

	agent := NewAgent(llm, buildTools(nil), 2)
	result, err := agent.Run(context.Background(), nil, "loop")
	if err == nil {
		t.Fatal("ожидалась ошибка лимита шагов")
	}
	llmSteps := 0
	toolSteps := 0
	for _, step := range result.Steps {
		if step.Type == "llm" {
			llmSteps++
		}
		if step.Type == "tool" {
			toolSteps++
		}
	}
	if llmSteps != 2 || toolSteps != 2 {
		t.Errorf("llm=%d tool=%d", llmSteps, toolSteps)
	}
}

func TestAgentUnknownTool(t *testing.T) {
	llm := &mockLLM{responses: []LLMResponse{
		{Message: LLMMessage{Role: "assistant", ToolCalls: []ToolCall{toolCall("c1", "unknown", `{}`)}}},
		{Message: LLMMessage{Role: "assistant", Content: "Не удалось вызвать инструмент."}},
	}}

	agent := NewAgent(llm, buildTools(nil), 3)
	result, err := agent.Run(context.Background(), nil, "test")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(result.Steps) != 3 {
		t.Errorf("steps = %+v", result.Steps)
	}
	foundToolError := false
	for _, step := range result.Steps {
		if step.Type == "tool" && step.Error != "" {
			foundToolError = true
		}
	}
	if !foundToolError {
		t.Errorf("steps = %+v", result.Steps)
	}
}

func TestDeepSeekToolCallRequest(t *testing.T) {
	var gotTools int
	var gotChoice, gotThinking string
	var gotTemp float64

	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req deepseekRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode: %v", err)
			return
		}
		gotTools = len(req.Tools)
		gotChoice = req.ToolChoice
		if req.Thinking != nil {
			gotThinking = req.Thinking.Type
		}
		if req.Temperature != nil {
			gotTemp = *req.Temperature
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{
				"message": map[string]any{
					"role":    "assistant",
					"content": nil,
					"tool_calls": []any{map[string]any{
						"id":   "c1",
						"type": "function",
						"function": map[string]string{
							"name":      "calculate",
							"arguments": `{"expression":"2+2"}`,
						},
					}},
				},
				"finish_reason": "tool_calls",
			}},
			"usage": map[string]any{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
		})
	}))
	defer up.Close()

	client := NewDeepSeekClient(up.URL, "test-key", "deepseek-flash")
	resp, err := client.Chat(context.Background(), []LLMMessage{{Role: "user", Content: "2+2"}}, buildToolsSchemas())
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if gotTools != 7 || gotChoice != "auto" || gotThinking != "disabled" || gotTemp != 0 {
		t.Errorf("tools=%d choice=%q thinking=%q temp=%v", gotTools, gotChoice, gotThinking, gotTemp)
	}
	if len(resp.Message.ToolCalls) != 1 || resp.Message.ToolCalls[0].Function.Name != "calculate" {
		t.Errorf("tool_calls = %+v", resp.Message.ToolCalls)
	}
	if resp.Usage.TotalTokens != 15 {
		t.Errorf("usage = %+v", resp.Usage)
	}
}

func TestGigaChatToolCallRequest(t *testing.T) {
	var got gigaRequest
	oauthCalls := 0

	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/oauth":
			oauthCalls++
			if r.Header.Get("RqUID") == "" {
				t.Error("OAuth без RqUID")
			}
			if !strings.HasPrefix(r.Header.Get("Authorization"), "Basic ") {
				t.Error("OAuth без Basic Authorization")
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"access_token": "test-access-token",
				"expires_at":   0,
			})
		case "/v1/chat/completions":
			if r.Header.Get("Authorization") != "Bearer test-access-token" {
				t.Errorf("Authorization = %q", r.Header.Get("Authorization"))
			}
			if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
				t.Errorf("decode: %v", err)
				return
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []any{map[string]any{
					"message": map[string]any{
						"role":               "assistant",
						"content":            nil,
						"functions_state_id": "state-1",
						"function_call": map[string]any{
							"name":      "calculate",
							"arguments": map[string]any{"expression": "2+2"},
						},
					},
					"finish_reason": "function_call",
				}},
				"usage": map[string]any{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer up.Close()

	client, err := NewGigaChatClient("GigaChat-2", "test-auth-key", "GIGACHAT_API_PERS", up.URL+"/oauth", up.URL, "")
	if err != nil {
		t.Fatal(err)
	}

	resp, err := client.Chat(context.Background(), []LLMMessage{{Role: "user", Content: "2+2"}}, buildToolsSchemas())
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if oauthCalls != 1 {
		t.Errorf("oauth calls = %d", oauthCalls)
	}
	if got.Model != "GigaChat-2" || got.FunctionCall != "auto" || len(got.Functions) != 7 {
		t.Errorf("request = %+v", got)
	}
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatalf("tool calls = %+v", resp.Message.ToolCalls)
	}
	if resp.Message.ToolCalls[0].Function.Name != "calculate" || !strings.Contains(resp.Message.ToolCalls[0].Function.Arguments, "2+2") {
		t.Errorf("tool call = %+v", resp.Message.ToolCalls[0])
	}
	if resp.Message.FunctionsStateID != "state-1" {
		t.Errorf("functions_state_id = %q", resp.Message.FunctionsStateID)
	}
	if resp.Usage.TotalTokens != 15 {
		t.Errorf("usage = %+v", resp.Usage)
	}
}

func TestDocumentStore(t *testing.T) {
	store, err := NewDocumentStore("documents/manifest.json")
	if err != nil {
		t.Fatal(err)
	}
	docs := store.List()
	if len(docs) != 7 {
		t.Fatalf("documents = %d", len(docs))
	}

	stats, err := store.Stats("book-100k")
	if err != nil {
		t.Fatal(err)
	}
	if stats.Chars == 0 || stats.Words == 0 || stats.Paragraphs == 0 || stats.Lines == 0 {
		t.Errorf("stats = %+v", stats)
	}

	search, err := store.Search("book-100k", "и", 3)
	if err != nil {
		t.Fatal(err)
	}
	if search.Count == 0 || len(search.Matches) != 3 {
		t.Errorf("search = %+v", search)
	}
	for _, match := range search.Matches {
		if strings.TrimSpace(match.Snippet) == "" {
			t.Errorf("empty snippet: %+v", match)
		}
	}
	if _, err := store.Search("book-100k", "   ", 3); err == nil {
		t.Error("ожидалась ошибка пустого запроса")
	}

	full, err := store.ReadAll("book-100k")
	if err != nil {
		t.Fatal(err)
	}
	if full.TotalChars == 0 || len([]rune(full.Text)) != full.TotalChars || full.Text == "" {
		t.Errorf("read all = id=%s total=%d textLen=%d", full.ID, full.TotalChars, len([]rune(full.Text)))
	}
}

func TestDocumentTools(t *testing.T) {
	store, err := NewDocumentStore("documents/manifest.json")
	if err != nil {
		t.Fatal(err)
	}
	tools := buildTools(store)

	value, err := tools["list_documents"].Run(context.Background(), json.RawMessage(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	if len(value.([]DocumentInfo)) != 7 {
		t.Errorf("list_documents = %+v", value)
	}

	value, err = tools["document_stats"].Run(context.Background(), json.RawMessage(`{"document_id":"book-100k"}`))
	if err != nil {
		t.Fatal(err)
	}
	if value.(DocumentStats).Chars == 0 {
		t.Errorf("document_stats = %+v", value)
	}

	value, err = tools["search_document"].Run(context.Background(), json.RawMessage(`{"document_id":"book-100k","query":"и","limit":2}`))
	if err != nil {
		t.Fatal(err)
	}
	if value.(DocumentSearchResult).Count == 0 {
		t.Errorf("search_document = %+v", value)
	}

	value, err = tools["read_document"].Run(context.Background(), json.RawMessage(`{"document_id":"book-100k"}`))
	if err != nil {
		t.Fatal(err)
	}
	text := value.(DocumentText)
	if text.TotalChars == 0 || len([]rune(text.Text)) != text.TotalChars {
		t.Errorf("read_document = id=%s total=%d textLen=%d", text.ID, text.TotalChars, len([]rune(text.Text)))
	}
}

func TestAttachAndReset(t *testing.T) {
	store, err := NewChatStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	docs, err := NewDocumentStore("documents/manifest.json")
	if err != nil {
		t.Fatal(err)
	}
	s := &server{cfg: config{contextLimit: 100000}, store: store, docs: docs}

	attach := func() *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodPost, "/api/documents/attach", strings.NewReader(`{"document_id":"book-100k"}`))
		rec := httptest.NewRecorder()
		s.attachDocument(rec, req)
		return rec
	}

	rec := attach()
	if rec.Code != http.StatusOK {
		t.Fatalf("attach status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var response struct {
		Document DocumentAttachment `json:"document"`
		Stats    struct {
			HistoryChars int `json:"historyChars"`
		} `json:"stats"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if response.Document.ID != "book-100k" || response.Stats.HistoryChars == 0 {
		t.Errorf("response = %+v", response)
	}

	if rec := attach(); rec.Code != http.StatusConflict {
		t.Errorf("duplicate attach status = %d", rec.Code)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/chat/reset", nil)
	rec = httptest.NewRecorder()
	s.reset(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("reset status = %d", rec.Code)
	}
	if len(store.Get().Documents) != 0 || len(store.Get().Messages) != 0 {
		t.Errorf("reset state = %+v", store.Get())
	}
}

func TestChatSendsWhenOverLimit(t *testing.T) {
	store, err := NewChatStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	state := ChatState{
		Messages: []LLMMessage{{Role: "user", Content: strings.Repeat("а", 500000)}},
		Turns:    []Turn{},
	}
	if err := store.Save(state); err != nil {
		t.Fatal(err)
	}

	llm := &mockLLM{responses: []LLMResponse{{
		Message: LLMMessage{Role: "assistant", Content: "ok"},
		Usage:   Usage{PromptTokens: 999999, CompletionTokens: 1, TotalTokens: 1000000},
	}}}
	s := &server{
		cfg:          config{contextLimit: 100},
		store:        store,
		agents:       map[string]*Agent{"mock": NewAgent(llm, buildTools(nil), 8)},
		defaultModel: "mock",
	}

	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"message":"привет"}`))
	rec := httptest.NewRecorder()
	s.chat(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if llm.calls != 1 {
		t.Errorf("llm calls = %d", llm.calls)
	}
	turns := store.Get().Turns
	if len(turns) != 1 || turns[0].Reply != "ok" {
		t.Errorf("turns = %+v", turns)
	}
}

func TestGigaChatToolResultRequest(t *testing.T) {
	client, err := NewGigaChatClient("GigaChat-2", "key", "GIGACHAT_API_PERS", "http://example.invalid/oauth", "http://example.invalid", "")
	if err != nil {
		t.Fatal(err)
	}

	messages := []LLMMessage{
		{Role: "user", Content: "2+2"},
		{
			Role:             "assistant",
			FunctionsStateID: "state-1",
			ToolCalls:        []ToolCall{toolCall("c1", "calculate", `{"expression":"2+2"}`)},
		},
		{Role: "tool", ToolCallID: "c1", Name: "calculate", Content: `{"ok":true,"result":4}`},
	}

	var request gigaRequest
	if err := json.Unmarshal([]byte(client.TraceRequest(messages, buildToolsSchemas())), &request); err != nil {
		t.Fatal(err)
	}
	if len(request.Messages) != 3 {
		t.Fatalf("messages = %d", len(request.Messages))
	}
	if request.Messages[1].FunctionCall == nil || request.Messages[1].FunctionCall.Name != "calculate" {
		t.Errorf("assistant function_call = %+v", request.Messages[1])
	}
	if request.Messages[1].FunctionsStateID != "state-1" {
		t.Errorf("assistant functions_state_id = %+v", request.Messages[1])
	}
	if request.Messages[2].Role != "function" || request.Messages[2].Name != "calculate" || !strings.Contains(request.Messages[2].Content, "result") {
		t.Errorf("function result = %+v", request.Messages[2])
	}
}

func buildToolsSchemas() []ToolSchema {
	agent := NewAgent(nil, buildTools(nil), 6)
	return agent.Tools()
}

func TestChatStore(t *testing.T) {
	store, err := NewChatStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	state := ChatState{Turns: []Turn{{ID: "1", User: "hi", Reply: "hello"}}}
	if err := store.Save(state); err != nil {
		t.Fatal(err)
	}
	got := store.Get()
	if len(got.Turns) != 1 {
		t.Fatalf("turns = %d", len(got.Turns))
	}
	history := got.History()
	if len(history) != 2 || history[0].Role != "user" || history[1].Role != "assistant" {
		t.Errorf("history = %+v", history)
	}
	if err := store.Reset(); err != nil {
		t.Fatal(err)
	}
	if len(store.Get().Turns) != 0 {
		t.Error("reset не очистил историю")
	}
}

func TestPersistenceWithToolCalls(t *testing.T) {
	dir := t.TempDir()
	store, err := NewChatStore(dir)
	if err != nil {
		t.Fatal(err)
	}

	state := ChatState{
		Version: 2,
		Messages: []LLMMessage{
			{Role: "user", Content: "Сколько будет 2+2?"},
			{Role: "assistant", ToolCalls: []ToolCall{toolCall("c1", "calculate", `{"expression":"2+2"}`)}},
			{Role: "tool", ToolCallID: "c1", Content: `{"ok":true,"result":4}`},
			{Role: "assistant", Content: "4"},
		},
		Turns: []Turn{{ID: "1", User: "Сколько будет 2+2?", Reply: "4"}},
	}
	if err := store.Save(state); err != nil {
		t.Fatal(err)
	}

	restarted, err := NewChatStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	got := restarted.Get()
	if len(got.Messages) != 4 {
		t.Fatalf("messages = %d", len(got.Messages))
	}
	history := got.History()
	if len(history) != 4 {
		t.Fatalf("history = %d", len(history))
	}
	if len(history[1].ToolCalls) != 1 || history[1].ToolCalls[0].Function.Name != "calculate" {
		t.Errorf("assistant tool call не восстановлен: %+v", history[1])
	}
	if history[2].Role != "tool" || history[2].ToolCallID != "c1" || !strings.Contains(history[2].Content, "4") {
		t.Errorf("tool result не восстановлен: %+v", history[2])
	}

	if err := restarted.Reset(); err != nil {
		t.Fatal(err)
	}
	afterReset := restarted.Get()
	if len(afterReset.Messages) != 0 || len(afterReset.Turns) != 0 {
		t.Errorf("reset не очистил контекст: %+v", afterReset)
	}
}
