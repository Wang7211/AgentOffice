import client from './client';

export interface LoginParams {
  username: string;
  password: string;
}

export interface RegisterParams {
  username: string;
  password: string;
  nickname?: string;
}

export interface UserInfo {
  user_id: number;
  username: string;
  nickname: string | null;
  role: string;
  avatar: string | null;
  status?: number;
  create_time?: string;
}

export interface LoginResult {
  access_token: string;
  token_type: string;
  user_id: number;
  username: string;
  nickname: string | null;
  role: string;
  avatar: string | null;
}

export async function login(params: LoginParams): Promise<LoginResult> {
  const res = await client.post('/auth/login', params);
  return res.data.data;
}

export async function register(params: RegisterParams): Promise<LoginResult> {
  const res = await client.post('/auth/register', params);
  return res.data.data;
}

export async function getMe(): Promise<UserInfo> {
  const res = await client.get('/auth/me');
  return res.data.data;
}
