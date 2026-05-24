import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Input,
  Button,
  Typography,
  Popconfirm,
  message,
  Empty,
  Tooltip,
  Dropdown,
} from 'antd';
import {
  SendOutlined,
  PlusOutlined,
  DeleteOutlined,
  EditOutlined,
  SearchOutlined,
  LogoutOutlined,
  SettingOutlined,
  RobotOutlined,
  UserOutlined,
  MessageOutlined,
  HistoryOutlined,
  BarsOutlined,
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../stores/authStore';
import { useChatStore } from '../stores/chatStore';
import {
  listSessions,
  getChatHistory,
  renameSession,
  deleteSession as deleteSessionApi,
  streamChat,
} from '../api/chat';
import type { ChatSession, ChatRecord } from '../api/chat';

const { Text } = Typography;
const PENDING_SESSION_PREFIX = '__pending_session__';

export default function Chat() {
  const navigate = useNavigate();
  const { user, logout, isAdmin } = useAuthStore();
  const {
    sessions,
    currentSessionId,
    messages,
    streaming,
    streamingContent,
    setSessions,
    setCurrentSession,
    setMessages,
    appendMessage,
    addStreamingChunk,
    clearStreaming,
    finishStreaming,
    setStreaming,
    removeSession,
    updateSessionName,
    addSession,
  } = useChatStore();

  const [inputValue, setInputValue] = useState('');
  const [searchKeyword, setSearchKeyword] = useState('');
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState('');
  const [loadingSessions, setLoadingSessions] = useState(false);
  const [sending, setSending] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const editInputRef = useRef<any>(null);
  const charQueueRef = useRef<string[]>([]);
  const typingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const streamCompletedRef = useRef(false);
  const pendingHistorySessionIdRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const loadSessions = useCallback(async () => {
    if (!user) return;
    setLoadingSessions(true);
    try {
      const data = await listSessions(user.user_id, searchKeyword || undefined);
      setSessions(data);
    } catch {
      // ignore
    } finally {
      setLoadingSessions(false);
    }
  }, [user, searchKeyword, setSessions]);

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  useEffect(() => {
    if (
      !currentSessionId ||
      streaming ||
      currentSessionId.startsWith(PENDING_SESSION_PREFIX)
    ) {
      return;
    }
    getChatHistory(currentSessionId)
      .then((records) => setMessages(currentSessionId, records))
      .catch(() => {});
  }, [currentSessionId, setMessages, streaming]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingContent]);

  // Typing queue timer: drains 1 char every 30ms for "逐字输出" effect
  // When stream ends, keeps running until queue is fully drained
  useEffect(() => {
    if (streaming) {
      typingTimerRef.current = setInterval(() => {
        const queue = charQueueRef.current;
        if (queue.length > 0) {
          const char = queue.shift()!;
          addStreamingChunk(char);
        } else if (streamCompletedRef.current) {
          // Queue empty AND stream done → cleanup
          const sessionId = pendingHistorySessionIdRef.current;
          pendingHistorySessionIdRef.current = null;
          streamCompletedRef.current = false;
          if (typingTimerRef.current) {
            clearInterval(typingTimerRef.current);
            typingTimerRef.current = null;
          }
          if (sessionId) {
            getChatHistory(sessionId)
              .then((records) => {
                finishStreaming(sessionId, records);
                loadSessions();
              })
              .catch(() => {
                finishStreaming();
                loadSessions();
              });
          } else {
            finishStreaming();
          }
        }
      }, 30);
    } else {
      if (typingTimerRef.current) {
        clearInterval(typingTimerRef.current);
        typingTimerRef.current = null;
      }
    }
    return () => {
      if (typingTimerRef.current) {
        clearInterval(typingTimerRef.current);
        typingTimerRef.current = null;
      }
    };
  }, [streaming, addStreamingChunk, finishStreaming, loadSessions]);

  const handleNewSession = () => {
    if (sending) return;
    setCurrentSession(null);
    clearStreaming();
    setInputValue('');
  };

  const handleSelectSession = (session: ChatSession) => {
    if (sending) return;
    setCurrentSession(session.session_id);
    clearStreaming();
  };

  const handleSend = async () => {
    const text = inputValue.trim();
    if (!text || sending || !user) return;

    setInputValue('');
    setSending(true);
    clearStreaming();
    charQueueRef.current = [];
    pendingHistorySessionIdRef.current = null;
    setStreaming(true);

    const localSessionId =
      currentSessionId || `${PENDING_SESSION_PREFIX}${Date.now()}`;
    const tempUserMsg: ChatRecord = {
      id: Date.now(),
      session_id: localSessionId,
      role: 'user',
      content: text,
      create_time: new Date().toISOString(),
    };

    let targetSessionId = currentSessionId;
    if (targetSessionId) {
      appendMessage(targetSessionId, tempUserMsg);
    } else {
      setCurrentSession(localSessionId);
      appendMessage(localSessionId, tempUserMsg);
    }

    let sessionCreated = false;

    abortRef.current = streamChat(
      text,
      targetSessionId,
      user.user_id,
      (content) => {
        // 逐字推入队列，由定时器逐字消费
        for (const char of content) {
          charQueueRef.current.push(char);
        }
      },
      (meta) => {
        if (meta.session_id && !sessionCreated) {
          sessionCreated = true;
          targetSessionId = meta.session_id;
          tempUserMsg.session_id = meta.session_id;
          if (currentSessionId !== meta.session_id) {
            setCurrentSession(meta.session_id);
            appendMessage(meta.session_id, tempUserMsg);
            loadSessions();
          }
        }
      },
      () => {
        setSending(false);
        // Mark stream completed; timer will drain the queue naturally
        pendingHistorySessionIdRef.current = targetSessionId;
        streamCompletedRef.current = true;
      },
      (err) => {
        setSending(false);
        streamCompletedRef.current = true;
        message.error('发送失败: ' + err.message);
      },
    );
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleRename = async (sessionId: string) => {
    if (!editName.trim()) return;
    try {
      await renameSession(sessionId, editName.trim());
      updateSessionName(sessionId, editName.trim());
      setEditingId(null);
      message.success('重命名成功');
    } catch {
      message.error('重命名失败');
    }
  };

  const handleDelete = async (sessionId: string) => {
    try {
      await deleteSessionApi(sessionId);
      removeSession(sessionId);
      message.success('会话已删除');
    } catch {
      message.error('删除失败');
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const startEditing = (session: ChatSession) => {
    setEditingId(session.session_id);
    setEditName(session.session_name || '');
    setTimeout(() => editInputRef.current?.focus(), 100);
  };

  const getInitial = (name?: string | null) => {
    return (name || 'U')[0].toUpperCase();
  };

  const quickPrompts = [
    'MCP 在智能体体系中解决了哪些问题？',
    '请解释 RAG 的工作方式。',
    '知识库分块有哪些常见策略？',
    '帮我整理一份会议纪要。',
  ];

  return (
    <div className="chat-layout">
      {/* ─── Sidebar ─── */}
      <div className="chat-sidebar">
        <div className="chat-sidebar-header">
          <div className="chat-sidebar-brand">
            <span className="brand-icon">
              <img src="/static/logo.jpg" className="logo-img" alt="AgentOffice" />
            </span>
            <span className="brand-copy">
              <strong>AgentOffice</strong>
            </span>
          </div>
        </div>

        <div className="chat-quick-start">
          <button className="quick-start-card" onClick={handleNewSession}>
            <span className="quick-start-icon">
              <PlusOutlined />
            </span>
            <span>
              <strong>新建对话</strong>
            </span>
          </button>
          {isAdmin() && (
            <Button
              className="quick-admin-button"
              type="default"
              icon={<SettingOutlined />}
              onClick={() => navigate('/admin')}
            >
              管理后台
            </Button>
          )}
        </div>

        <div className="chat-search">
          <Input
            prefix={<SearchOutlined style={{ color: 'var(--gray-400)' }} />}
            placeholder="搜索对话..."
            value={searchKeyword}
            onChange={(e) => setSearchKeyword(e.target.value)}
            allowClear
            style={{ height: 40 }}
          />
        </div>

        <div className="chat-sessions">
          {sessions.length === 0 && !loadingSessions && (
            <div style={{ padding: '40px 16px' }}>
              <Empty
                image={<MessageOutlined style={{ fontSize: 48, color: 'var(--gray-300)' }} />}
                description={<span style={{ color: 'var(--gray-400)' }}>暂无对话</span>}
              />
            </div>
          )}
          {sessions.length > 0 && (
            <div className="session-group-label">最近对话</div>
          )}
          {sessions.map((session) => (
            <div
              key={session.session_id}
              className={`session-item${currentSessionId === session.session_id ? ' active' : ''}`}
              onClick={() => handleSelectSession(session)}
            >
              {editingId === session.session_id ? (
                <Input
                  ref={editInputRef}
                  size="small"
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  onPressEnter={() => handleRename(session.session_id)}
                  onBlur={() => handleRename(session.session_id)}
                  onClick={(e) => e.stopPropagation()}
                  style={{ width: '100%', height: 32, fontSize: 13 }}
                />
              ) : (
                <>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, flex: 1, minWidth: 0 }}>
                    <HistoryOutlined style={{ color: 'var(--gray-300)', fontSize: 14, flexShrink: 0 }} />
                    <Text
                      ellipsis
                      className="session-name"
                    >
                      {session.session_name || '新对话'}
                    </Text>
                  </div>
                  <div className="session-actions" onClick={(e) => e.stopPropagation()}>
                    <Tooltip title="重命名">
                      <Button
                        type="text"
                        size="small"
                        icon={<EditOutlined />}
                        onClick={() => startEditing(session)}
                        style={{ width: 28, height: 28, minWidth: 28 }}
                      />
                    </Tooltip>
                    <Popconfirm
                      title="删除此会话？"
                      description="删除后不可恢复"
                      onConfirm={() => handleDelete(session.session_id)}
                      okText="删除"
                      cancelText="取消"
                    >
                      <Tooltip title="删除">
                        <Button
                          type="text"
                          size="small"
                          danger
                          icon={<DeleteOutlined />}
                          style={{ width: 28, height: 28, minWidth: 28 }}
                        />
                      </Tooltip>
                    </Popconfirm>
                  </div>
                </>
              )}
            </div>
          ))}
        </div>

        {/* User footer */}
        <div className="chat-sidebar-footer">
          <div className="user-info">
            <div className="user-avatar">
              {user?.avatar ? <img src={user.avatar} className="avatar-img" alt="avatar" /> : getInitial(user?.nickname || user?.username)}
            </div>
            <div>
              <div className="user-name">{user?.nickname || user?.username}</div>
              <div className="user-role">
                {user?.role === 'admin' ? '管理员' : '用户'}
              </div>
            </div>
          </div>
          <Dropdown
            menu={{
              items: [
                ...(isAdmin()
                  ? [{
                      key: 'admin',
                      icon: <SettingOutlined />,
                      label: '后台管理',
                      onClick: () => navigate('/admin'),
                    }]
                  : []),
                {
                  key: 'logout',
                  icon: <LogoutOutlined />,
                  label: '退出登录',
                  danger: true,
                  onClick: handleLogout,
                },
              ],
            }}
            placement="topRight"
            trigger={['click']}
          >
            <Button
              type="text"
              icon={<BarsOutlined />}
              style={{ width: 32, height: 32, minWidth: 32, color: 'var(--gray-400)' }}
            />
          </Dropdown>
        </div>
      </div>

      {/* ─── Main Chat Area ─── */}
      <div className="chat-main">
        {currentSessionId || messages.length > 0 ? (
          <>
            <div className="chat-header">
              <div className="chat-header-title">
                <span className="chat-header-dot" />
                {sessions.find((s) => s.session_id === currentSessionId)?.session_name || '新对话'}
              </div>
              <Text type="secondary" style={{ fontSize: 12 }}>
                {messages.filter((m) => m.role === 'assistant').length} 条回复
              </Text>
            </div>

            <div className="chat-messages">
              {messages.map((msg) => (
                <div key={msg.id} className={`message-row ${msg.role}`}>
                  <div
                    className={`message-avatar ${msg.role}`}
                  >
                    {msg.role === 'user' ? (user?.avatar ? <img src={user.avatar} className="avatar-img" alt="avatar" /> : getInitial(user?.nickname)) : <RobotOutlined />}
                  </div>
                  <div>
                    <div
                      className="message-bubble"
                    >
                      {msg.content}
                    </div>
                    <div
                      style={{
                        fontSize: 11,
                        color: 'var(--gray-400)',
                        marginTop: 4,
                        paddingLeft: msg.role === 'assistant' ? 4 : 0,
                        paddingRight: msg.role === 'user' ? 4 : 0,
                        textAlign: msg.role === 'user' ? 'right' : 'left',
                      }}
                    >
                      {new Date(msg.create_time).toLocaleTimeString('zh-CN', {
                        hour: '2-digit',
                        minute: '2-digit',
                      })}
                    </div>
                  </div>
                </div>
              ))}

              {/* Streaming */}
              {streaming && (
                <div className="message-row assistant">
                  <div className="message-avatar assistant">
                    <RobotOutlined />
                  </div>
                  <div>
                    <div className="message-bubble streaming">
                      {streamingContent || <div className="typing-indicator"><span /><span /><span /></div>}
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            <div className="chat-input-area">
              <div className="chat-input-wrapper">
                <Input.TextArea
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="输入您的问题，按 Enter 发送..."
                  autoSize={{ minRows: 1, maxRows: 4 }}
                  disabled={sending}
                  variant="borderless"
                />
                <div className="chat-input-actions">
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {sending ? '正在回复...' : ''}
                  </Text>
                  <Button
                    type="primary"
                    icon={<SendOutlined />}
                    onClick={handleSend}
                    loading={sending}
                    disabled={!inputValue.trim()}
                    shape="circle"
                    style={{
                      width: 40,
                      height: 40,
                      minWidth: 40,
                      boxShadow: inputValue.trim()
                        ? '0 2px 8px rgba(79, 110, 247, 0.3)'
                        : 'none',
                    }}
                  />
                </div>
              </div>
            </div>
          </>
        ) : (
          <>
            <div className="chat-header">
              <div className="chat-header-title">
                <span className="chat-header-dot" />
                新对话
              </div>
            </div>

            <div className="chat-empty-home">
              <div className="empty-center">
                <div className="empty-logo">
                  <img src="/static/logo.jpg" className="logo-img" alt="AgentOffice" />
                </div>
                <h1>有什么可以帮忙的？</h1>
                <div className="prompt-grid">
                  {quickPrompts.map((prompt) => (
                    <button
                      key={prompt}
                      className="prompt-card"
                      onClick={() => setInputValue(prompt)}
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="chat-input-area">
              <div className="chat-input-wrapper">
                <Input.TextArea
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="输入你的问题..."
                  autoSize={{ minRows: 1, maxRows: 4 }}
                  disabled={sending}
                  variant="borderless"
                />
                <div className="chat-input-actions">
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {sending ? '正在回复...' : ''}
                  </Text>
                  <Button
                    type="primary"
                    icon={<SendOutlined />}
                    onClick={handleSend}
                    loading={sending}
                    disabled={!inputValue.trim()}
                    shape="circle"
                    style={{ width: 40, height: 40, minWidth: 40 }}
                  />
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
