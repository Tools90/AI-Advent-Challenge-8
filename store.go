package main

import (
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

type DocumentAttachment struct {
	ID         string    `json:"id"`
	Title      string    `json:"title"`
	Author     string    `json:"author"`
	Chars      int       `json:"chars"`
	AttachedAt time.Time `json:"attachedAt"`
}

type TokenStats struct {
	HistoryChars         int     `json:"historyChars"`
	RequestChars         int     `json:"requestChars"`
	RequestTokens        int     `json:"requestTokens"`
	ResponseTokens       int     `json:"responseTokens"`
	LastPromptTokens     int     `json:"lastPromptTokens"`
	LastCompletionTokens int     `json:"lastCompletionTokens"`
	MaxPromptTokens      int     `json:"maxPromptTokens"`
	TotalTokens          int     `json:"totalTokens"`
	CumulativeTokens     int     `json:"cumulativeTokens"`
	ContextLimit         int     `json:"contextLimit"`
	ContextPercent       float64 `json:"contextPercent"`
	CostUSD              float64 `json:"costUsd"`
}

type Turn struct {
	ID         string      `json:"id"`
	Model      string      `json:"model,omitempty"`
	User       string      `json:"user"`
	Reply      string      `json:"reply,omitempty"`
	Steps      []TraceStep `json:"steps,omitempty"`
	Usage      Usage       `json:"usage"`
	TokenStats TokenStats  `json:"tokenStats"`
	LatencyMs  int64       `json:"latencyMs"`
	Calls      int         `json:"calls"`
	Error      string      `json:"error,omitempty"`
	CreatedAt  time.Time   `json:"createdAt"`
}

type ChatState struct {
	Version   int                  `json:"version,omitempty"`
	Messages  []LLMMessage         `json:"messages,omitempty"`
	Documents []DocumentAttachment `json:"documents,omitempty"`
	Turns     []Turn               `json:"turns"`
}

func (s ChatState) History() []LLMMessage {
	if len(s.Messages) > 0 {
		return append([]LLMMessage(nil), s.Messages...)
	}

	// Migration from the old format, where only final messages were stored.
	out := make([]LLMMessage, 0, len(s.Turns)*2)
	for _, turn := range s.Turns {
		out = append(out, LLMMessage{Role: "user", Content: turn.User})
		if turn.Reply != "" {
			out = append(out, LLMMessage{Role: "assistant", Content: turn.Reply})
		}
	}
	return out
}

func (s *ChatState) EnsureMessages() {
	if len(s.Messages) == 0 && len(s.Turns) > 0 {
		s.Messages = s.History()
	}
}

type ChatStore struct {
	mu    sync.RWMutex
	path  string
	state ChatState
}

func NewChatStore(dir string) (*ChatStore, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("создание %s: %w", dir, err)
	}

	store := &ChatStore{path: filepath.Join(dir, "chat.json")}
	data, err := os.ReadFile(store.path)
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			return nil, fmt.Errorf("чтение %s: %w", store.path, err)
		}
		return store, nil
	}
	if len(data) > 0 {
		if err := json.Unmarshal(data, &store.state); err != nil {
			return nil, fmt.Errorf("разбор %s: %w", store.path, err)
		}
	}
	return store, nil
}

func (s *ChatStore) Get() ChatState {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return clone(s.state)
}

func (s *ChatStore) Save(state ChatState) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state = clone(state)
	return s.saveLocked()
}

func (s *ChatStore) Reset() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state = ChatState{Turns: []Turn{}, Documents: []DocumentAttachment{}}
	return s.saveLocked()
}

func (s *ChatStore) saveLocked() error {
	data, err := json.MarshalIndent(s.state, "", "  ")
	if err != nil {
		return err
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	_ = os.Remove(s.path)
	return os.Rename(tmp, s.path)
}

func newID() string {
	buf := make([]byte, 8)
	if _, err := rand.Read(buf); err != nil {
		return fmt.Sprintf("%d", time.Now().UnixNano())
	}
	return fmt.Sprintf("%x", buf)
}

func clone[T any](v T) T {
	data, err := json.Marshal(v)
	if err != nil {
		return v
	}
	var out T
	if err := json.Unmarshal(data, &out); err != nil {
		return v
	}
	return out
}
