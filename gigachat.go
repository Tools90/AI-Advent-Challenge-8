package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"
)

type GigaChatClient struct {
	model    string
	authKey  string
	scope    string
	oauthURL string
	baseURL  string
	client   *http.Client

	mu        sync.Mutex
	token     string
	expiresAt time.Time
}

type gigaFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type gigaFunctionCall struct {
	Name      string `json:"name"`
	Arguments any    `json:"arguments"`
}

type gigaMessage struct {
	Role             string            `json:"role"`
	Content          string            `json:"content"`
	Name             string            `json:"name,omitempty"`
	FunctionCall     *gigaFunctionCall `json:"function_call,omitempty"`
	FunctionsStateID string            `json:"functions_state_id,omitempty"`
}

type gigaRequest struct {
	Model        string         `json:"model"`
	Messages     []gigaMessage  `json:"messages"`
	Functions    []gigaFunction `json:"functions,omitempty"`
	FunctionCall string         `json:"function_call,omitempty"`
	Temperature  *float64       `json:"temperature,omitempty"`
}

type gigaUsage struct {
	PromptTokens          int `json:"prompt_tokens"`
	CompletionTokens      int `json:"completion_tokens"`
	TotalTokens           int `json:"total_tokens"`
	PrecachedPromptTokens int `json:"precached_prompt_tokens"`
}

type gigaResponse struct {
	Choices []struct {
		Message      gigaMessage `json:"message"`
		FinishReason string      `json:"finish_reason"`
	} `json:"choices"`
	Model string    `json:"model"`
	Usage gigaUsage `json:"usage"`
}

func NewGigaChatClient(model, authKey, scope, oauthURL, baseURL, caBundle string) (*GigaChatClient, error) {
	transport := http.DefaultTransport.(*http.Transport).Clone()

	if strings.TrimSpace(caBundle) != "" {
		pool, err := x509.SystemCertPool()
		if err != nil || pool == nil {
			pool = x509.NewCertPool()
		}
		data, err := os.ReadFile(caBundle)
		if err != nil {
			return nil, fmt.Errorf("чтение CA-бандла %s: %w", caBundle, err)
		}
		if !pool.AppendCertsFromPEM(data) {
			return nil, fmt.Errorf("в %s нет валидных PEM-сертификатов", caBundle)
		}
		transport.TLSClientConfig = &tls.Config{RootCAs: pool, MinVersion: tls.VersionTLS12}
	}

	return &GigaChatClient{
		model:    model,
		authKey:  strings.TrimSpace(authKey),
		scope:    scope,
		oauthURL: oauthURL,
		baseURL:  strings.TrimRight(baseURL, "/"),
		client:   &http.Client{Timeout: 180 * time.Second, Transport: transport},
	}, nil
}

func (g *GigaChatClient) Model() string { return g.model }

func (g *GigaChatClient) TraceRequest(messages []LLMMessage, tools []ToolSchema) string {
	req, err := g.buildRequest(messages, tools)
	if err != nil {
		return err.Error()
	}
	encoded, err := json.Marshal(req)
	if err != nil {
		return err.Error()
	}
	return string(encoded)
}

func (g *GigaChatClient) Chat(ctx context.Context, messages []LLMMessage, tools []ToolSchema) (LLMResponse, error) {
	reqBody, err := g.buildRequest(messages, tools)
	if err != nil {
		return LLMResponse{}, err
	}

	for attempt := 0; attempt < 2; attempt++ {
		token, err := g.accessToken(ctx, attempt > 0)
		if err != nil {
			return LLMResponse{}, err
		}

		response, status, err := g.call(ctx, token, reqBody)
		if err != nil {
			return LLMResponse{}, err
		}
		if status == http.StatusUnauthorized && attempt == 0 {
			continue
		}
		if status != http.StatusOK {
			return LLMResponse{}, fmt.Errorf("GigaChat вернул статус %d", status)
		}
		return response, nil
	}

	return LLMResponse{}, errors.New("GigaChat: не удалось авторизоваться")
}

func (g *GigaChatClient) buildRequest(messages []LLMMessage, tools []ToolSchema) (gigaRequest, error) {
	converted := make([]gigaMessage, 0, len(messages))
	for _, message := range messages {
		item := gigaMessage{
			Role:             message.Role,
			Content:          message.Content,
			Name:             message.Name,
			FunctionsStateID: message.FunctionsStateID,
		}

		if message.Role == "assistant" && len(message.ToolCalls) > 0 {
			call := message.ToolCalls[0]
			var args any = map[string]any{}
			if strings.TrimSpace(call.Function.Arguments) != "" {
				if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
					args = map[string]any{}
				}
			}
			item.FunctionCall = &gigaFunctionCall{
				Name:      call.Function.Name,
				Arguments: args,
			}
		}

		if message.Role == "tool" {
			item.Role = "function"
			item.Name = message.Name
		}

		converted = append(converted, item)
	}

	functions := make([]gigaFunction, 0, len(tools))
	for _, tool := range tools {
		functions = append(functions, gigaFunction{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
			Parameters:  tool.Function.Parameters,
		})
	}

	temp := 0.0
	return gigaRequest{
		Model:        g.model,
		Messages:     converted,
		Functions:    functions,
		FunctionCall: "auto",
		Temperature:  &temp,
	}, nil
}

func (g *GigaChatClient) call(ctx context.Context, token string, reqBody gigaRequest) (LLMResponse, int, error) {
	body, err := json.Marshal(reqBody)
	if err != nil {
		return LLMResponse{}, 0, fmt.Errorf("сборка запроса GigaChat: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, g.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return LLMResponse{}, 0, fmt.Errorf("создание запроса GigaChat: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)

	resp, err := g.client.Do(req)
	if err != nil {
		return LLMResponse{}, 0, fmt.Errorf("GigaChat: %w", err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return LLMResponse{}, resp.StatusCode, fmt.Errorf("чтение ответа GigaChat: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		if resp.StatusCode == http.StatusUnauthorized {
			return LLMResponse{}, resp.StatusCode, nil
		}
		return LLMResponse{}, resp.StatusCode, gigaHTTPError(resp.StatusCode, data)
	}

	var out gigaResponse
	if err := json.Unmarshal(data, &out); err != nil {
		return LLMResponse{}, resp.StatusCode, fmt.Errorf("разбор ответа GigaChat: %w", err)
	}
	if len(out.Choices) == 0 {
		return LLMResponse{}, resp.StatusCode, errors.New("GigaChat вернул пустой choices")
	}

	choice := out.Choices[0]
	message := LLMMessage{
		Role:             "assistant",
		Content:          choice.Message.Content,
		FunctionsStateID: choice.Message.FunctionsStateID,
	}

	if call := choice.Message.FunctionCall; call != nil {
		var arguments []byte
		switch value := call.Arguments.(type) {
		case string:
			if strings.TrimSpace(value) == "" {
				arguments = []byte("{}")
			} else {
				arguments = []byte(value)
			}
		default:
			arguments, _ = json.Marshal(value)
		}
		message.ToolCalls = []ToolCall{{
			ID:   newUUID(),
			Type: "function",
			Function: struct {
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
			}{
				Name:      call.Name,
				Arguments: string(arguments),
			},
		}}
	}

	return LLMResponse{
		Message:      message,
		FinishReason: choice.FinishReason,
		Usage: Usage{
			PromptTokens:     out.Usage.PromptTokens,
			CompletionTokens: out.Usage.CompletionTokens,
			TotalTokens:      out.Usage.TotalTokens,
		},
	}, resp.StatusCode, nil
}

func (g *GigaChatClient) accessToken(ctx context.Context, force bool) (string, error) {
	g.mu.Lock()
	defer g.mu.Unlock()

	if !force && g.token != "" && time.Now().Before(g.expiresAt.Add(-1*time.Minute)) {
		return g.token, nil
	}

	form := url.Values{"scope": {g.scope}}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, g.oauthURL, strings.NewReader(form.Encode()))
	if err != nil {
		return "", fmt.Errorf("создание OAuth-запроса GigaChat: %w", err)
	}
	req.Header.Set("Authorization", "Basic "+g.authKey)
	req.Header.Set("RqUID", newUUID())
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	resp, err := g.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("OAuth GigaChat: %w", err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return "", fmt.Errorf("чтение OAuth-ответа GigaChat: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return "", gigaHTTPError(resp.StatusCode, data)
	}

	var out struct {
		AccessToken string `json:"access_token"`
		ExpiresAt   int64  `json:"expires_at"`
	}
	if err := json.Unmarshal(data, &out); err != nil {
		return "", fmt.Errorf("разбор OAuth-ответа GigaChat: %w", err)
	}
	if out.AccessToken == "" {
		return "", errors.New("GigaChat OAuth не вернул access_token")
	}

	g.token = out.AccessToken
	if out.ExpiresAt > 0 {
		if out.ExpiresAt > 1_000_000_000_000 {
			g.expiresAt = time.UnixMilli(out.ExpiresAt)
		} else {
			g.expiresAt = time.Unix(out.ExpiresAt, 0)
		}
	} else {
		g.expiresAt = time.Now().Add(25 * time.Minute)
	}
	return g.token, nil
}

func gigaHTTPError(status int, data []byte) error {
	var payload struct {
		Message string `json:"message"`
		Error   struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(data, &payload) == nil {
		if payload.Message != "" {
			return fmt.Errorf("GigaChat: %s", payload.Message)
		}
		if payload.Error.Message != "" {
			return fmt.Errorf("GigaChat: %s", payload.Error.Message)
		}
	}
	return fmt.Errorf("GigaChat вернул статус %d", status)
}

func newUUID() string {
	buf := make([]byte, 16)
	_, _ = rand.Read(buf)
	buf[6] = (buf[6] & 0x0f) | 0x40
	buf[8] = (buf[8] & 0x3f) | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", buf[0:4], buf[4:6], buf[6:8], buf[8:10], buf[10:16])
}
