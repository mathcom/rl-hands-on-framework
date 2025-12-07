import React, { useState, useEffect } from 'react';
import { 
  REQUIREMENTS_TXT, 
  AGENT_TEMPLATE_PY, 
  MAIN_PY, 
  RUN_GUIDE_MD, 
  APP_TITLE 
} from './constants';
import CodeBlock from './components/CodeBlock';
import ChatAssistant from './components/ChatAssistant';

enum Tab {
  GUIDE = 'Guide',
  REQUIREMENTS = 'requirements.txt',
  AGENT = 'agent_template.py',
  MAIN = 'main.py'
}

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<Tab>(Tab.GUIDE);
  // 초기 로딩 시 화면 크기에 따라 채팅창 열림 여부 결정
  const [showChat, setShowChat] = useState(() => window.innerWidth >= 768);
  const [isMobile, setIsMobile] = useState(() => window.innerWidth < 768);

  // 화면 크기 감지 로직 개선
  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      
      // [수정 포인트] Android 키보드 이슈 해결
      // 모바일 상태에서는 resize(키보드 등장)가 일어나도 showChat 상태를 건드리지 않음
      // 데스크탑으로 커졌을 때만 채팅창을 강제로 복구함
      if (!mobile) {
        setShowChat(true);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const renderContent = () => {
    switch (activeTab) {
      case Tab.GUIDE:
        return (
          <div className="prose prose-invert max-w-none bg-slate-900/50 p-6 md:p-8 rounded-lg border border-slate-700/50 shadow-inner overflow-auto h-full pb-24 md:pb-8">
            <h2 className="text-xl md:text-2xl font-bold mb-4 text-purple-400">🌕 Lunar Lander Challenge</h2>
            <div className="space-y-4 text-slate-300">
              <p className="text-sm md:text-base">
                강화학습 실습에 오신 것을 환영합니다! <span className="text-yellow-400 font-mono">LunarLander-v2</span> 환경에서 
                우주선을 안전하게 착륙시키세요.
              </p>
              <div className="bg-slate-800 p-4 rounded border-l-4 border-purple-500 my-4">
                <h3 className="font-bold text-white mb-2 text-sm md:text-base">🚀 목표 (Objective)</h3>
                <ul className="list-disc list-inside space-y-1 text-xs md:text-sm">
                  <li><strong>State (8):</strong> 위치, 속도, 각도 등</li>
                  <li><strong>Action (4):</strong> 엔진 조절 (0~3)</li>
                  <li><strong>Reward:</strong> 안전 착륙 시 +200점</li>
                </ul>
              </div>
              <hr className="border-slate-700 my-6"/>
              <div className="whitespace-pre-wrap font-sans text-xs md:text-sm leading-6">
                {RUN_GUIDE_MD.trim()}
              </div>
            </div>
          </div>
        );
      case Tab.REQUIREMENTS:
        return <CodeBlock code={REQUIREMENTS_TXT} filename="requirements.txt" />;
      case Tab.AGENT:
        return <CodeBlock code={AGENT_TEMPLATE_PY} filename="agent_template.py" />;
      case Tab.MAIN:
        return <CodeBlock code={MAIN_PY} filename="main.py" />;
      default:
        return null;
    }
  };

  return (
    // [수정] h-screen -> h-[100dvh] : 모바일 브라우저 주소창 대응
    <div className="flex flex-col h-[100dvh] bg-slate-950 text-slate-200 overflow-hidden font-sans">
      {/* Header */}
      <header className="flex-none h-14 md:h-16 bg-slate-900 border-b border-slate-800 flex items-center justify-between px-4 md:px-6 shadow-md z-20">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-lg flex items-center justify-center font-bold text-white shadow-lg shadow-purple-500/20 text-sm">
            RL
          </div>
          <h1 className="text-base md:text-lg font-bold tracking-tight text-white truncate max-w-[200px] md:max-w-none">
            {APP_TITLE}
          </h1>
        </div>
        
        {/* Desktop Chat Toggle */}
        <button 
          onClick={() => setShowChat(!showChat)}
          className="hidden md:block px-4 py-2 rounded-md text-sm font-medium transition-all bg-slate-800 text-slate-400 hover:text-white border border-slate-700"
        >
          {showChat ? 'Hide Assistant' : 'Show Assistant'}
        </button>

        {/* Mobile Chat Button (Icon) */}
        <button 
          onClick={() => setShowChat(true)}
          className="md:hidden p-2 text-purple-400 hover:text-white"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" /></svg>
        </button>
      </header>

      {/* Main Layout */}
      <div className="flex-grow flex overflow-hidden relative">
        {/* Desktop Sidebar */}
        <nav className="hidden md:flex flex-none w-64 bg-slate-900 border-r border-slate-800 flex-col p-4 gap-2">
          <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2 px-2">Resources</div>
          {Object.values(Tab).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`text-left px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200 flex items-center gap-3 ${
                activeTab === tab
                  ? 'bg-purple-600 text-white shadow-lg shadow-purple-900/50'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-white'
              }`}
            >
              {tab}
            </button>
          ))}
        </nav>

        {/* Content Area */}
        <main className="flex-grow flex flex-col relative w-full">
          <div className="flex-grow overflow-hidden p-4 md:p-6">
             {renderContent()}
          </div>
        </main>
        
        {/* Desktop Chat Panel */}
        <aside className={`hidden md:block w-[400px] bg-slate-900 border-l border-slate-800 transition-all duration-300 ${showChat ? 'translate-x-0' : 'translate-x-full absolute right-0 h-full'}`}>
           {showChat && <ChatAssistant onClose={() => setShowChat(false)} isMobile={false} />}
        </aside>
      </div>

      {/* Mobile Bottom Navigation */}
      <nav className="md:hidden flex-none h-16 bg-slate-900 border-t border-slate-800 flex justify-around items-center px-2 z-10 pb-safe">
        {Object.values(Tab).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`flex flex-col items-center justify-center w-full h-full gap-1 ${
              activeTab === tab ? 'text-purple-400' : 'text-slate-500'
            }`}
          >
            {tab === Tab.GUIDE ? (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" /></svg>
            ) : (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" /></svg>
            )}
            <span className="text-[10px] font-medium truncate max-w-[60px]">{tab === Tab.GUIDE ? '가이드' : tab.split('.')[0]}</span>
          </button>
        ))}
      </nav>

      {/* Mobile Chat Overlay */}
      {isMobile && showChat && (
        <div className="fixed inset-0 z-50 bg-slate-900 flex flex-col animate-in slide-in-from-bottom duration-300">
          <div className="flex-none h-14 border-b border-slate-700 flex items-center justify-between px-4 bg-slate-800">
            <h2 className="font-bold text-white">AI Assistant</h2>
            <button onClick={() => setShowChat(false)} className="text-slate-400 hover:text-white p-2">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
            </button>
          </div>
          <div className="flex-grow overflow-hidden">
            <ChatAssistant onClose={() => setShowChat(false)} isMobile={true} />
          </div>
        </div>
      )}
    </div>
  );
};

export default App;