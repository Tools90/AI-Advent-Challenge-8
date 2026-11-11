package main

import (
	"bufio"
	"context"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
	"unicode/utf8"
)

//go:embed web/index.html
var webFS embed.FS

type config struct {
	authKeyFile     string
	authKey         string
	scope           string
	oauthURL        string
	baseURL         string
	model           string
	caBundle        string
	deepseekKeyFile string
	deepseekKey     string
	deepseekBaseURL string
	deepseekModel   string
	contextLimit    int
	addr            string
}

type ModelOption struct {
	ID         string `json:"id"`
	Label      string `json:"label"`
	Model      string `json:"model"`
	Configured bool   `json:"configured"`
}

type server struct {
	cfg          config
	store        *ChatStore
	docs         *DocumentStore
	agents       map[string]*Agent
	models       []ModelOption
	defaultModel string
	chatMu       sync.Mutex
}

func (c config) configured() bool { return c.authKey != "" }

func main() {
	if err := loadDotEnv(".env"); err != nil {
		log.Printf("config: %v", err)
	}

	cfg := loadConfig()
	if !cfg.configured() {
		log.Printf("gigachat: Authorization Key не задан в %s", cfg.authKeyFile)
	}

	store, err := NewChatStore("data")
	if err != nil {
		log.Fatalf("store: %v", err)
	}

	docs, err := NewDocumentStore("documents/manifest.json")
	if err != nil {
		log.Fatalf("documents: %v", err)
	}

	caBundle := cfg.caBundle
	if _, err := os.Stat(caBundle); err != nil {
		log.Printf("gigachat: CA-бандл %s недоступен, использую системные сертификаты: %v", caBundle, err)
		caBundle = ""
	}

	client, err := NewGigaChatClient(cfg.model, cfg.authKey, cfg.scope, cfg.oauthURL, cfg.baseURL, caBundle)
	if err != nil {
		log.Fatalf("gigachat: %v", err)
	}

	tools := buildTools(docs)
	gigaAgent := NewAgent(client, tools, 8)
	deepseekClient := NewDeepSeekClient(cfg.deepseekBaseURL, cfg.deepseekKey, cfg.deepseekModel)
	deepseekAgent := NewAgent(deepseekClient, tools, 8)

	models := []ModelOption{
		{ID: "deepseek", Label: "DeepSeek Flash", Model: cfg.deepseekModel, Configured: cfg.deepseekKey != ""},
		{ID: "gigachat", Label: "GigaChat 2 Lite", Model: cfg.model, Configured: cfg.configured()},
	}

	s := &server{
		cfg:          cfg,
		store:        store,
		docs:         docs,
		agents:       map[string]*Agent{"gigachat": gigaAgent, "deepseek": deepseekAgent},
		models:       models,
		defaultModel: "deepseek",
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /", s.index)
	mux.HandleFunc("GET /healthz", s.health)
	mux.HandleFunc("GET /api/state", s.state)
	mux.HandleFunc("GET /api/documents", s.documents)
	mux.HandleFunc("POST /api/documents/attach", s.attachDocument)
	mux.HandleFunc("POST /api/chat", s.chat)
	mux.HandleFunc("POST /api/chat/reset", s.reset)

	srv := &http.Server{
		Addr:              cfg.addr,
		Handler:           logRequests(mux),
		ReadHeaderTimeout: 10 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go func() {
		log.Printf("сервер запущен: http://localhost%s (модель: %s)", cfg.addr, cfg.model)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("server: %v", err)
		}
	}()

	<-ctx.Done()
	log.Println("останавливаюсь...")

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Printf("shutdown: %v", err)
	}
}

func (s *server) index(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	data, err := webFS.ReadFile("web/index.html")
	if err != nil {
		http.Error(w, "index.html не найден", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(data)
}

func (s *server) health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"status":       "ok",
		"model":        s.cfg.model,
		"models":       s.models,
		"defaultModel": s.defaultModel,
		"configured":   s.cfg.configured(),
		"contextLimit": s.cfg.contextLimit,
	})
}

type chatStats struct {
	HistoryChars        int     `json:"historyChars"`
	ApproxHistoryTokens int     `json:"approxHistoryTokens"`
	CumulativeTokens    int     `json:"cumulativeTokens"`
	ContextLimit        int     `json:"contextLimit"`
	ContextPercent      float64 `json:"contextPercent"`
	AttachedDocuments   int     `json:"attachedDocuments"`
}

func (s *server) chatStats(state ChatState) chatStats {
	chars := messagesChars(state.Messages)
	return chatStats{
		HistoryChars:        chars,
		ApproxHistoryTokens: approxTokens(chars),
		CumulativeTokens:    cumulativeTokens(state.Turns),
		ContextLimit:        s.cfg.contextLimit,
		ContextPercent:      contextPercent(approxTokens(chars), s.cfg.contextLimit),
		AttachedDocuments:   len(state.Documents),
	}
}

func (s *server) state(w http.ResponseWriter, r *http.Request) {
	state := s.store.Get()
	state.EnsureMessages()
	writeJSON(w, http.StatusOK, map[string]any{
		"turns":             state.Turns,
		"tools":             s.agents[s.defaultModel].Tools(),
		"model":             s.cfg.model,
		"models":            s.models,
		"defaultModel":      s.defaultModel,
		"configured":        s.cfg.configured(),
		"contextLimit":      s.cfg.contextLimit,
		"documents":         s.docs.List(),
		"attachedDocuments": state.Documents,
		"stats":             s.chatStats(state),
	})
}

func (s *server) documents(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, s.docs.List())
}

func (s *server) attachDocument(w http.ResponseWriter, r *http.Request) {
	var body struct {
		DocumentID string `json:"document_id"`
	}
	if err := decodeJSON(w, r, &body); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	id := strings.TrimSpace(body.DocumentID)
	if id == "" {
		writeError(w, http.StatusBadRequest, "document_id пустой")
		return
	}

	s.chatMu.Lock()
	defer s.chatMu.Unlock()

	state := s.store.Get()
	state.EnsureMessages()

	for _, attached := range state.Documents {
		if attached.ID == id {
			writeError(w, http.StatusConflict, "документ уже добавлен в контекст")
			return
		}
	}

	doc, ok := s.docs.Get(id)
	if !ok {
		writeError(w, http.StatusNotFound, "документ не найден")
		return
	}
	text, err := s.docs.Text(id)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	content := fmt.Sprintf("Документ для анализа.\nID: %s\nНазвание: %s\nАвтор: %s\n\n%s", doc.ID, doc.Title, doc.Author, text)
	attachment := DocumentAttachment{
		ID:         doc.ID,
		Title:      doc.Title,
		Author:     doc.Author,
		Chars:      utf8.RuneCountInString(text),
		AttachedAt: time.Now(),
	}

	state.Messages = append(state.Messages, LLMMessage{Role: "user", Content: content})
	state.Documents = append(state.Documents, attachment)
	state.Version = 3
	if err := s.store.Save(state); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	stats := s.chatStats(state)
	writeJSON(w, http.StatusOK, map[string]any{
		"document":     attachment,
		"stats":        stats,
		"contextLimit": s.cfg.contextLimit,
	})
}

func (s *server) chat(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Message string `json:"message"`
		Model   string `json:"model"`
	}
	if err := decodeJSON(w, r, &body); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	message := strings.TrimSpace(body.Message)
	if message == "" {
		writeError(w, http.StatusBadRequest, "сообщение пустое")
		return
	}
	if len([]rune(message)) > 4000 {
		writeError(w, http.StatusBadRequest, "сообщение слишком длинное (максимум 4000 символов)")
		return
	}

	modelID := strings.TrimSpace(body.Model)
	if modelID == "" {
		modelID = s.defaultModel
	}
	agent, ok := s.agents[modelID]
	if !ok {
		writeError(w, http.StatusBadRequest, "неизвестная модель")
		return
	}

	s.chatMu.Lock()
	defer s.chatMu.Unlock()

	state := s.store.Get()
	state.EnsureMessages()
	history := append([]LLMMessage(nil), state.Messages...)

	histChars := messagesChars(history)
	requestChars := histChars + utf8.RuneCountInString(agentSystemPrompt) + utf8.RuneCountInString(message)
	estimatedTokens := approxTokens(requestChars)
	if estimatedTokens > s.cfg.contextLimit {
		log.Printf("context: оценка %d токенов > лимита %d, отправляю запрос без блокировки", estimatedTokens, s.cfg.contextLimit)
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Minute)
	defer cancel()

	result, err := agent.Run(ctx, history, message)

	turn := Turn{
		ID:        newID(),
		Model:     modelID,
		User:      message,
		Reply:     result.Reply,
		Steps:     result.Steps,
		Usage:     result.Usage,
		LatencyMs: result.LatencyMs,
		Calls:     result.Calls,
		CreatedAt: time.Now(),
	}
	turn.TokenStats = TokenStats{
		HistoryChars:         histChars,
		RequestChars:         requestChars,
		RequestTokens:        result.Usage.PromptTokens,
		ResponseTokens:       result.Usage.CompletionTokens,
		LastPromptTokens:     result.LastPromptTokens,
		LastCompletionTokens: result.LastCompletionTokens,
		MaxPromptTokens:      result.MaxPromptTokens,
		TotalTokens:          result.Usage.TotalTokens,
		CumulativeTokens:     cumulativeTokens(state.Turns) + result.Usage.TotalTokens,
		ContextLimit:         s.cfg.contextLimit,
		ContextPercent:       contextPercent(result.MaxPromptTokens, s.cfg.contextLimit),
		CostUSD:              0,
	}
	if turn.TokenStats.ContextPercent == 0 {
		turn.TokenStats.ContextPercent = contextPercent(estimatedTokens, s.cfg.contextLimit)
	}
	if err != nil {
		turn.Error = err.Error()
	}

	state.Messages = append(state.Messages, result.Messages...)
	state.Version = 3
	state.Turns = append(state.Turns, turn)
	if saveErr := s.store.Save(state); saveErr != nil {
		writeError(w, http.StatusInternalServerError, saveErr.Error())
		return
	}

	writeJSON(w, http.StatusOK, turn)
}

func (s *server) reset(w http.ResponseWriter, r *http.Request) {
	s.chatMu.Lock()
	defer s.chatMu.Unlock()

	if err := s.store.Reset(); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func loadConfig() config {
	port := envOr("PORT", "8087")

	contextLimit := 128000
	if raw := strings.TrimSpace(os.Getenv("CONTEXT_LIMIT")); raw != "" {
		if value, err := strconv.Atoi(raw); err == nil && value > 0 {
			contextLimit = value
		}
	}

	authKeyFile := envOr("GIGACHAT_AUTH_FILE", envOr("GIGACHAT_AUTH_KEY_FILE", "gigachat_auth.txt"))
	deepseekKeyFile := envOr("DEEPSEEK_API_KEY_FILE", "api_key.txt")
	return config{
		authKeyFile:     authKeyFile,
		authKey:         readSecret(authKeyFile, "your-gigachat-authorization-key"),
		scope:           envOr("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
		oauthURL:        envOr("GIGACHAT_OAUTH_URL", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"),
		baseURL:         strings.TrimRight(envOr("GIGACHAT_BASE_URL", "https://api.giga.chat"), "/"),
		model:           envOr("GIGACHAT_MODEL", "GigaChat-2"),
		caBundle:        envOr("GIGACHAT_CA_BUNDLE", "certs/russian_trusted_ca_bundle.pem"),
		deepseekKeyFile: deepseekKeyFile,
		deepseekKey:     readSecret(deepseekKeyFile, "sk-your-key-here"),
		deepseekBaseURL: strings.TrimRight(envOr("DEEPSEEK_BASE_URL", "https://api.deepseek.com"), "/"),
		deepseekModel:   envOr("DEEPSEEK_MODEL", "deepseek-flash"),
		contextLimit:    contextLimit,
		addr:            ":" + port,
	}
}

func readSecret(path, placeholder string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	value := unquote(strings.TrimSpace(strings.TrimPrefix(string(data), "\ufeff")))
	if value == placeholder || strings.Contains(value, "your-") || strings.Contains(value, "your-authorization-key") || strings.Contains(value, "your-api-key") {
		return ""
	}
	return value
}

func approxTokens(chars int) int {
	if chars <= 0 {
		return 0
	}
	return (chars + 2) / 3
}

func messageChars(message LLMMessage) int {
	chars := utf8.RuneCountInString(message.Content) + utf8.RuneCountInString(message.Name)
	for _, call := range message.ToolCalls {
		chars += utf8.RuneCountInString(call.Function.Name)
		chars += utf8.RuneCountInString(call.Function.Arguments)
	}
	return chars
}

func messagesChars(messages []LLMMessage) int {
	total := 0
	for _, message := range messages {
		total += messageChars(message)
	}
	return total
}

func cumulativeTokens(turns []Turn) int {
	total := 0
	for _, turn := range turns {
		total += turn.Usage.TotalTokens
	}
	return total
}

func contextPercent(tokens, limit int) float64 {
	if limit <= 0 {
		return 0
	}
	return float64(tokens) / float64(limit) * 100
}

func envOr(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func decodeJSON(w http.ResponseWriter, r *http.Request, dst any) error {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	if err := json.NewDecoder(r.Body).Decode(dst); err != nil {
		return fmt.Errorf("некорректный JSON: %w", err)
	}
	return nil
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start).Round(time.Millisecond))
	})
}

func loadDotEnv(path string) error {
	f, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("файл %s не найден, использую переменные окружения", path)
		}
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(strings.TrimPrefix(scanner.Text(), "\ufeff"))
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(strings.TrimPrefix(key, "export "))
		value = unquote(strings.TrimSpace(value))
		if key == "" {
			continue
		}
		if _, exists := os.LookupEnv(key); !exists {
			_ = os.Setenv(key, value)
		}
	}
	return scanner.Err()
}

func unquote(v string) string {
	if len(v) >= 2 {
		if (v[0] == '"' && v[len(v)-1] == '"') || (v[0] == '\'' && v[len(v)-1] == '\'') {
			return v[1 : len(v)-1]
		}
	}
	return v
}
