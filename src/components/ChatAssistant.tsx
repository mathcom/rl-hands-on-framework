import React, { useState, useRef, useEffect } from 'react';

// 이제 파일 내용을 import 할 필요가 없습니다! (백엔드가 읽음)

interface Message {
  id: string;
  role: 'user' | 'model';
  text: string;
}

interface ChatAssistantProps {
  onClose?: () => void;
  isMobile: boolean;
}

const ChatAssistant: React.FC<ChatAssistantProps> = ({ onClose, isMobile }) => {
  const [messages, setMessages] = useState<Message[]>([
    { 
      id: 'init', 
      role: 'model', 
      text: '안녕하세요! 강화학습 RAG 조교입니다. 프로젝트 코드에 대해 무엇이든 물어보세요.' 
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMsg: Message = { id: Date.now().toString(), role: 'user', text: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      // [핵심 변경] 복잡한 프롬프트 없이 질문만 백엔드로 보냅니다.
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: input,
          // 필요하다면 대화 내역도 보낼 수 있습니다.
          history: [] 
        })
      });

      if (!response.ok) throw new Error('Backend Server Error');
      
      const data = await response.json();
      const aiResponseText = data.reply || "답변을 가져오지 못했습니다.";
      
      setMessages(prev => [...prev, { id: (Date.now()+1).toString(), role: 'model', text: aiResponseText }]);

    } catch (error) {
      console.error(error);
      setMessages(prev => [...prev, { id: Date.now().toString(), role: 'model', text: '⚠️ 백엔드 연결 실패. 잠시 후 다시 시도해주세요.' }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`flex flex-col h-full bg-slate-800 ${!isMobile && 'rounded-lg border border-slate-700 shadow-lg'} overflow-hidden`}>
      {!isMobile && (
        <div className="p-4 bg-slate-900 border-b border-slate-700 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
            <div>
              <h2 className="font-semibold text-white text-sm">RAG AI Assistant</h2>
              <span className="text-xs text-slate-400">Powered by LangChain & Chroma</span>
            </div>
          </div>
          {onClose && (
            <button onClick={onClose} className="text-slate-400 hover:text-white md:hidden">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
            </button>
          )}
        </div>
      )}
      
      <div className="flex-grow overflow-y-auto p-4 space-y-4 bg-slate-900/50">
        {messages.map((msg) => (
          <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] md:max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap shadow-sm ${
              msg.role === 'user' 
                ? 'bg-purple-600 text-white rounded-br-none' 
                : 'bg-slate-700 text-slate-200 rounded-bl-none border border-slate-600'
            }`}>
              {msg.text}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-slate-700 rounded-2xl rounded-bl-none px-4 py-3 flex gap-1 items-center">
               <span className="text-xs text-slate-400 mr-2">Searching Docs...</span>
               <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce"></div>
               <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce delay-75"></div>
               <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce delay-150"></div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="p-3 md:p-4 bg-slate-800 border-t border-slate-700">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
            placeholder="코드에 대해 물어보세요..."
            className="flex-grow bg-slate-900 border border-slate-600 text-white text-base rounded-full px-4 py-2 focus:outline-none focus:ring-2 focus:ring-purple-500 placeholder-slate-500"
          />
          <button
            onClick={handleSend}
            disabled={loading || !input.trim()}
            className="bg-purple-600 hover:bg-purple-500 disabled:bg-slate-600 text-white w-10 h-10 rounded-full flex items-center justify-center transition-colors shadow-lg flex-none"
          >
            <svg className="w-5 h-5 translate-x-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" /></svg>
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatAssistant;