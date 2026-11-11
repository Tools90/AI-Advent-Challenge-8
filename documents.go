package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode/utf8"
)

type DocumentInfo struct {
	ID           string `json:"id"`
	Title        string `json:"title"`
	Author       string `json:"author"`
	File         string `json:"file"`
	Size         string `json:"size"`
	Chars        int    `json:"chars"`
	ApproxTokens int    `json:"approxTokens"`
	Description  string `json:"description,omitempty"`
}

type documentManifest struct {
	Documents []DocumentInfo `json:"documents"`
}

type DocumentStore struct {
	dir  string
	docs map[string]DocumentInfo
	list []DocumentInfo
}

type DocumentStats struct {
	ID         string `json:"id"`
	Chars      int    `json:"chars"`
	Words      int    `json:"words"`
	Paragraphs int    `json:"paragraphs"`
	Lines      int    `json:"lines"`
}

type SearchMatch struct {
	Position int    `json:"position"`
	Snippet  string `json:"snippet"`
}

type DocumentSearchResult struct {
	ID      string        `json:"id"`
	Query   string        `json:"query"`
	Count   int           `json:"count"`
	Matches []SearchMatch `json:"matches"`
}

type DocumentText struct {
	ID         string `json:"id"`
	Title      string `json:"title"`
	TotalChars int    `json:"totalChars"`
	Text       string `json:"text"`
}

func NewDocumentStore(manifestPath string) (*DocumentStore, error) {
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		return nil, fmt.Errorf("чтение %s: %w", manifestPath, err)
	}

	var manifest documentManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return nil, fmt.Errorf("разбор %s: %w", manifestPath, err)
	}
	if len(manifest.Documents) == 0 {
		return nil, fmt.Errorf("в %s нет документов", manifestPath)
	}

	dir := filepath.Dir(manifestPath)
	store := &DocumentStore{
		dir:  dir,
		docs: map[string]DocumentInfo{},
	}

	for _, doc := range manifest.Documents {
		doc.ID = strings.TrimSpace(doc.ID)
		doc.Title = strings.TrimSpace(doc.Title)
		doc.File = strings.TrimSpace(doc.File)
		if doc.ID == "" || doc.Title == "" || doc.File == "" {
			return nil, fmt.Errorf("документ с пустым id/title/file")
		}
		if _, exists := store.docs[doc.ID]; exists {
			return nil, fmt.Errorf("повторяющийся document id %q", doc.ID)
		}
		if _, err := os.Stat(filepath.Join(dir, doc.File)); err != nil {
			return nil, fmt.Errorf("документ %q: %w", doc.ID, err)
		}
		store.docs[doc.ID] = doc
		store.list = append(store.list, doc)
	}

	return store, nil
}

func (s *DocumentStore) List() []DocumentInfo {
	out := make([]DocumentInfo, len(s.list))
	copy(out, s.list)
	return out
}

func (s *DocumentStore) Get(id string) (DocumentInfo, bool) {
	doc, ok := s.docs[id]
	return doc, ok
}

func (s *DocumentStore) Text(id string) (string, error) {
	doc, ok := s.docs[id]
	if !ok {
		return "", fmt.Errorf("документ %q не найден", id)
	}
	data, err := os.ReadFile(filepath.Join(s.dir, doc.File))
	if err != nil {
		return "", fmt.Errorf("чтение %s: %w", doc.File, err)
	}
	text := strings.TrimSpace(strings.TrimPrefix(string(data), "\ufeff"))
	return strings.ReplaceAll(text, "\r\n", "\n"), nil
}

func (s *DocumentStore) Stats(id string) (DocumentStats, error) {
	text, err := s.Text(id)
	if err != nil {
		return DocumentStats{}, err
	}

	paragraphs := 0
	for _, part := range strings.Split(text, "\n\n") {
		if strings.TrimSpace(part) != "" {
			paragraphs++
		}
	}
	lines := 0
	for _, line := range strings.Split(text, "\n") {
		if strings.TrimSpace(line) != "" {
			lines++
		}
	}

	return DocumentStats{
		ID:         id,
		Chars:      utf8.RuneCountInString(text),
		Words:      len(strings.Fields(text)),
		Paragraphs: paragraphs,
		Lines:      lines,
	}, nil
}

func (s *DocumentStore) Search(id, query string, limit int) (DocumentSearchResult, error) {
	text, err := s.Text(id)
	if err != nil {
		return DocumentSearchResult{}, err
	}
	query = strings.TrimSpace(query)
	if query == "" {
		return DocumentSearchResult{}, fmt.Errorf("пустой поисковый запрос")
	}
	if limit <= 0 {
		limit = 5
	}

	textRunes := []rune(text)
	lowerText := []rune(strings.ToLower(text))
	queryRunes := []rune(strings.ToLower(query))

	result := DocumentSearchResult{ID: id, Query: query, Matches: []SearchMatch{}}
	for i := 0; i+len(queryRunes) <= len(lowerText); {
		if !runeSliceEqual(lowerText[i:i+len(queryRunes)], queryRunes) {
			i++
			continue
		}

		result.Count++
		if len(result.Matches) < limit {
			const radius = 80
			start := i - radius
			if start < 0 {
				start = 0
			}
			end := i + len(queryRunes) + radius
			if end > len(textRunes) {
				end = len(textRunes)
			}
			result.Matches = append(result.Matches, SearchMatch{
				Position: i,
				Snippet:  strings.TrimSpace(string(textRunes[start:end])),
			})
		}
		i += len(queryRunes)
	}

	return result, nil
}

func (s *DocumentStore) ReadAll(id string) (DocumentText, error) {
	doc, ok := s.docs[id]
	if !ok {
		return DocumentText{}, fmt.Errorf("документ %q не найден", id)
	}
	text, err := s.Text(id)
	if err != nil {
		return DocumentText{}, err
	}

	return DocumentText{
		ID:         id,
		Title:      doc.Title,
		TotalChars: utf8.RuneCountInString(text),
		Text:       text,
	}, nil
}

func runeSliceEqual(a, b []rune) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func (s *DocumentStore) IDs() []string {
	ids := make([]string, 0, len(s.docs))
	for id := range s.docs {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}
