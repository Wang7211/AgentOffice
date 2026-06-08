import client from './client';

export interface DashboardStats {
  user_count: number;
  session_count: number;
  file_count: number;
  chunk_count: number;
  tool_call_count: number;
  tool_success_count?: number;
  tool_failure_count?: number;
  avg_tool_cost?: number;
  p95_tool_cost?: number;
  today_chat_count: number;
  hourly_activity?: {
    label: string;
    chat_count: number;
    tool_count: number;
    success_count: number;
    failure_count: number;
  }[];
  recent_sessions: { session_id: string; session_name: string; create_time: string }[];
  recent_tools: { id: number; tool_name: string; status: number; cost_time: number; create_time: string }[];
}

export interface UserItem {
  id: number;
  username: string;
  nickname: string | null;
  role: string;
  avatar: string | null;
  status: number;
  create_time: string;
}

export interface PaginatedResult<T> {
  total: number;
  page: number;
  page_size: number;
  items: T[];
}

export interface KnowledgeFileItem {
  id: number;
  file_name: string;
  file_suffix: string;
  file_size: number;
  chunk_count: number;
  create_time: string;
}

export interface ChunkItem {
  id: number;
  chunk_index: number;
  chunk_text: string;
  vector_id: string;
  create_time: string;
}

export interface TraceItem {
  id: number;
  chat_record_id: number;
  tool_name: string;
  tool_input: string;
  tool_result: string;
  cost_time: number;
  status: number;
  error_msg: string;
  create_time: string;
  session_name: string | null;
}

export interface TraceDetail extends TraceItem {
  session_id: string | null;
  user_message: string | null;
  assistant_message: string | null;
}

export interface ConfigItem {
  id: number;
  config_key: string;
  config_value: string;
  config_name: string;
  config_type: string;
  remark: string | null;
}

// Dashboard
export async function getDashboard(): Promise<DashboardStats> {
  const res = await client.get('/admin/dashboard');
  return res.data.data;
}

// Users
export async function listUsers(
  page: number,
  pageSize: number,
  keyword?: string,
): Promise<PaginatedResult<UserItem>> {
  const params: any = { page, page_size: pageSize };
  if (keyword) params.keyword = keyword;
  const res = await client.get('/admin/users', { params });
  return res.data.data;
}

export async function updateUser(
  userId: number,
  data: { nickname?: string; role?: string; status?: number; password?: string },
): Promise<void> {
  await client.put(`/admin/users/${userId}`, data);
}

export async function deleteUser(userId: number): Promise<void> {
  await client.delete(`/admin/users/${userId}`);
}

// Knowledge Files
export async function listKnowledgeFiles(
  page: number,
  pageSize: number,
): Promise<PaginatedResult<KnowledgeFileItem>> {
  const res = await client.get('/admin/knowledge/files', {
    params: { page, page_size: pageSize },
  });
  return res.data.data;
}

export async function deleteKnowledgeFile(fileId: number): Promise<void> {
  await client.delete(`/admin/knowledge/files/${fileId}`);
}

export async function getFileChunks(
  fileId: number,
  page: number,
  pageSize: number,
): Promise<PaginatedResult<ChunkItem> & { file_name: string }> {
  const res = await client.get(`/admin/knowledge/files/${fileId}/chunks`, {
    params: { page, page_size: pageSize },
  });
  return res.data.data;
}

// Traces
export async function listTraces(
  page: number,
  pageSize: number,
  toolName?: string,
  status?: number,
): Promise<PaginatedResult<TraceItem>> {
  const params: any = { page, page_size: pageSize };
  if (toolName) params.tool_name = toolName;
  if (status !== undefined) params.status = status;
  const res = await client.get('/admin/traces', { params });
  return res.data.data;
}

export async function getTraceDetail(traceId: number): Promise<TraceDetail> {
  const res = await client.get(`/admin/traces/${traceId}`);
  return res.data.data;
}

// Config
export async function listConfig(): Promise<ConfigItem[]> {
  const res = await client.get('/admin/config');
  return res.data.data;
}

export async function updateConfig(id: number, configValue: string): Promise<void> {
  await client.put('/admin/config', { id, config_value: configValue });
}

// Knowledge upload
export async function uploadKnowledge(file: File): Promise<any> {
  const formData = new FormData();
  formData.append('upload_file', file);
  const res = await client.post('/knowledge/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000,
  });
  return res.data.data;
}
