import { create } from 'zustand';
import type { ChatSession, ChatRecord } from '../api/chat';

interface ChatState {
  sessions: ChatSession[];
  currentSessionId: string | null;
  messages: ChatRecord[];
  messagesMap: Record<string, ChatRecord[]>;
  loading: boolean;
  streaming: boolean;
  streamingContent: string;
  setSessions: (sessions: ChatSession[]) => void;
  setCurrentSession: (sessionId: string | null) => void;
  setMessages: (sessionId: string, messages: ChatRecord[]) => void;
  appendMessage: (sessionId: string, message: ChatRecord) => void;
  addStreamingChunk: (chunk: string) => void;
  clearStreaming: () => void;
  finishStreaming: (sessionId?: string, messages?: ChatRecord[]) => void;
  setLoading: (loading: boolean) => void;
  setStreaming: (streaming: boolean) => void;
  removeSession: (sessionId: string) => void;
  updateSessionName: (sessionId: string, name: string) => void;
  addSession: (session: ChatSession) => void;
}

export const useChatStore = create<ChatState>()((set, get) => ({
  sessions: [],
  currentSessionId: null,
  messages: [],
  messagesMap: {},
  loading: false,
  streaming: false,
  streamingContent: '',

  setSessions: (sessions) => set({ sessions }),

  setCurrentSession: (sessionId) => {
    const state = get();
    const cached = sessionId ? state.messagesMap[sessionId] : [];
    set({
      currentSessionId: sessionId,
      messages: cached || [],
      streamingContent: '',
    });
  },

  setMessages: (sessionId, messages) => {
    set((state) => ({
      messagesMap: { ...state.messagesMap, [sessionId]: messages },
      messages: state.currentSessionId === sessionId ? messages : state.messages,
    }));
  },

  appendMessage: (sessionId, message) => {
    set((state) => {
      const existing = state.messagesMap[sessionId] || [];
      const updated = [...existing, message];
      return {
        messagesMap: { ...state.messagesMap, [sessionId]: updated },
        messages:
          state.currentSessionId === sessionId ? updated : state.messages,
      };
    });
  },

  addStreamingChunk: (chunk) =>
    set((state) => ({ streamingContent: state.streamingContent + chunk })),

  clearStreaming: () => set({ streamingContent: '' }),

  finishStreaming: (sessionId, messages) => {
    set((state) => {
      const nextState: Partial<ChatState> = {
        streaming: false,
        streamingContent: '',
      };

      if (sessionId && messages) {
        nextState.messagesMap = {
          ...state.messagesMap,
          [sessionId]: messages,
        };
        nextState.messages =
          state.currentSessionId === sessionId ? messages : state.messages;
      }

      return nextState;
    });
  },

  setLoading: (loading) => set({ loading }),

  setStreaming: (streaming) => set({ streaming }),

  removeSession: (sessionId) => {
    set((state) => {
      const sessions = state.sessions.filter((s) => s.session_id !== sessionId);
      const messagesMap = { ...state.messagesMap };
      delete messagesMap[sessionId];
      return {
        sessions,
        messagesMap,
        currentSessionId:
          state.currentSessionId === sessionId ? null : state.currentSessionId,
        messages:
          state.currentSessionId === sessionId ? [] : state.messages,
      };
    });
  },

  updateSessionName: (sessionId, name) => {
    set((state) => ({
      sessions: state.sessions.map((s) =>
        s.session_id === sessionId ? { ...s, session_name: name } : s,
      ),
    }));
  },

  addSession: (session) => {
    set((state) => ({
      sessions: [session, ...state.sessions],
    }));
  },
}));
