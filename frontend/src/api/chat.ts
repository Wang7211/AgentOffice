import client from './client';

export interface ChatSession {
  session_id: string;
  session_name: string;
  model_name: string;
  create_time: string;
  update_time: string;
}

export interface ChatRecord {
  id: number;
  session_id: string;
  role: 'user' | 'assistant';
  content: string;
  create_time: string;
}

export interface Citation {
  file_name: string;
}

export interface ChatResult {
  session_id: string;
  message_id: number;
  answer: string;
  tool_name?: string;
  tool_result?: string;
  tool_calls?: any[];
  plan?: any[];
  reflection?: any;
  citations?: Citation[];
}

export async function listSessions(
  userId: number,
  keyword?: string,
): Promise<ChatSession[]> {
  const params: any = { user_id: userId };
  if (keyword) params.keyword = keyword;
  const res = await client.get('/chat/sessions', { params });
  return res.data.data;
}

export async function getChatHistory(
  sessionId?: string,
): Promise<ChatRecord[]> {
  const params: any = {};
  if (sessionId) params.session_id = sessionId;
  const res = await client.get('/chat/history', { params });
  return res.data.data;
}

export async function renameSession(
  sessionId: string,
  sessionName: string,
): Promise<ChatSession> {
  const res = await client.put(`/chat/sessions/${sessionId}/rename`, {
    session_name: sessionName,
  });
  return res.data.data;
}

export async function deleteSession(sessionId: string): Promise<void> {
  await client.delete(`/chat/sessions/${sessionId}`);
}

// SSE streaming chat
export function streamChat(
  message: string,
  sessionId: string | null,
  userId: number,
  onMessage: (content: string) => void,
  onMeta: (data: any) => void,
  onDone: (data?: any) => void,
  onError: (error: Error) => void,
): AbortController {
  const controller = new AbortController();

  fetch('/api/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${localStorage.getItem('token')}`,
    },
    body: JSON.stringify({
      message,
      session_id: sessionId,
      stream: true,
      user_id: userId,
    }),
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const reader = response.body?.getReader();
      if (!reader) throw new Error('No response body');

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        let currentEvent = '';
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7).trim();
            continue;
          }
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6);
            try {
              const data = JSON.parse(dataStr);
              if (currentEvent === 'meta') {
                onMeta(data);
              } else if (currentEvent === 'message') {
                onMessage(data.content || '');
              } else if (currentEvent === 'done') {
                onDone(data);
              }
            } catch {
              // ignore parse errors for partial chunks
            }
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== 'AbortError') {
        onError(err);
      }
    });

  return controller;
}
