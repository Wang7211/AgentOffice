import { useEffect } from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuthStore } from '../stores/authStore';
import { getMe } from '../api/auth';

interface Props {
  children: React.ReactNode;
  requireAdmin?: boolean;
}

export default function ProtectedRoute({ children, requireAdmin }: Props) {
  const { token, user, setAuth } = useAuthStore();
  const location = useLocation();

  useEffect(() => {
    if (token && user) {
      getMe().then((fresh) => setAuth(token, fresh)).catch(() => {});
    }
  }, []);

  if (!token || !user) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  if (requireAdmin && user.role !== 'admin') {
    return <Navigate to="/chat" replace />;
  }

  return <>{children}</>;
}
