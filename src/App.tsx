import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactMarkdown from 'react-markdown'; // [추가] 마크다운 렌더러
import { 
  REQUIREMENTS_TXT, 
  LEVEL1_CODE, 
  LEVEL2_CODE, 
  LEVEL3_CODE,
  MAIN_PY, 
  RUN_GUIDE_MD, 
  APP_TITLE 
} from './constants';
import CodeBlock from './components/CodeBlock';
import ChatAssistant from './components/ChatAssistant';

enum Level {
  LV1 = 'Level 1: Tabular (Q-Learning)',
  LV2 = 'Level 2: Value-based (DQN)',
  LV3 = 'Level 3: Policy-based (PPO)'
}

enum Tab {
  GUIDE = 'Guide',
  REQUIREMENTS = 'requirements.txt',
  AGENT = 'agent.py (Template)',
  MAIN = 'main.py'
}

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<Tab>(Tab.GUIDE);
  const [currentLevel, setCurrentLevel] = useState<Level>(Level.LV1);
  const [isMobile, setIsMobile] = useState(() => window.innerWidth < 768);
  const [showChat, setShowChat] = useState(() => window.innerWidth >= 768);
  const [chatWidth, setChatWidth] = useState(400);
  const [isResizing, setIsResizing] = useState(false);
  const sidebarRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      if (!mobile && !showChat) {
         // 데스크탑 전환 시 로직
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [showChat]);

  const startResizing = useCallback(() => { setIsResizing(true); }, []);
  const stopResizing = useCallback(() => { setIsResizing(false); }, []);

  const resize = useCallback((mouseMoveEvent: MouseEvent) => {
    if (isResizing) {
      const newWidth = window.innerWidth - mouseMoveEvent.clientX;
      if (newWidth > 300 && newWidth < 800) {
        setChatWidth(newWidth);
      }
    }
  }, [isResizing]);

  useEffect(() => {
    window.addEventListener('mousemove', resize);
    window.addEventListener('mouseup', stopResizing);
    return () => {
      window.removeEventListener('mousemove', resize);
      window.removeEventListener('mouseup', stopResizing);
    };
  }, [resize, stopResizing]);

  const getAgentCode = () => {
    switch (currentLevel) {
      case Level.LV1: return LEVEL1_CODE;
      case Level.LV2: return LEVEL2_CODE;
      case Level.LV3: return LEVEL3_CODE;
      default: return LEVEL1_CODE;
    }
  };

  const getAgentFilename = () => {
    switch (currentLevel) {
      case Level.LV1: return 'agent_tabular.py';
      case Level.LV2: return 'agent_dqn.py';
      case Level.LV3: return 'agent_ppo.py';
      default: return 'agent.py';
    }
  };

  const renderContent = () => {
    switch (activeTab) {
      case Tab.GUIDE:
        return (
          <div className="prose prose-invert max-w-none bg-slate-900/50 p-6 md:p-8 rounded-lg border border-slate-700/50 shadow-inner overflow-auto h-full pb-24 md:pb-8">
            <h2 className="text-xl md:text-2xl font-bold mb-4 text-purple-400">🌕 Lunar Lander Challenge</h2>
            
            {/* 현재 레벨 요약 박스 */}
            <div className="bg-blue-900/20 border border-blue-500/30 rounded-lg p-4 mb-6">
              <span className="text-xs font-bold text-blue-400 uppercase tracking-wider">Current Mission</span>
              <h3 className="text-lg font-bold text-white mt-1 mb-2">{currentLevel}</h3>
              <p className="text-sm text-slate-300 m-0">
                {currentLevel === Level.LV1 && "연속적인 상태를 이산화(Discretize)하여 Q-Table을 완성하세요."}
                {currentLevel === Level.LV2 && "신경망(Neural Network)을 이용해 Q-Function을 근사하세요."}
                {currentLevel === Level.LV3 && "확률적 정책(Policy)을 직접 최적화하여 부드러운 제어를 구현하세요."}
              </p>
            </div>

            {/* [수정] ReactMarkdown 적용 및 Tailwind 스타일 매핑 */}
            <div className="text-slate-300">
              <ReactMarkdown
                components={{
                  h1: ({node, ...props}) => <h1 className="text-2xl font-bold text-purple-400 mt-8 mb-4 border-b border-slate-700 pb-2" {...props} />,
                  h2: ({node, ...props}) => <h2 className="text-xl font-bold text-white mt-6 mb-3" {...props} />,
                  h3: ({node, ...props}) => <h3 className="text-lg font-bold text-blue-300 mt-5 mb-2" {...props} />,
                  p: ({node, ...props}) => <p className="mb-4 leading-relaxed text-sm md:text-base" {...props} />,
                  ul: ({node, ...props}) => <ul className="list-disc list-inside space-y-1 mb-4 ml-2" {...props} />,
                  li: ({node, ...props}) => <li className="text-slate-300" {...props} />,
                  code: ({node, className, children, ...props}: any) => {
                    const match = /language-(\w+)/.exec(className || '');
                    const isInline = !match && !String(children).includes('\n');
                    return isInline ? (
                      <code className="bg-slate-800 text-yellow-400 px-1.5 py-0.5 rounded font-mono text-xs md:text-sm" {...props}>
                        {children}
                      </code>
                    ) : (
                      <div className="bg-slate-950 rounded-lg p-4 my-4 border border-slate-800 overflow-x-auto">
                        <code className="font-mono text-xs md:text-sm text-slate-300 block" {...props}>
                          {children}
                        </code>
                      </div>
                    );
                  },
                  a: ({node, ...props}) => <a className="text-blue-400 hover:underline" target="_blank" rel="noopener noreferrer" {...props} />,
                  blockquote: ({node, ...props}) => <blockquote className="border-l-4 border-slate-600 pl-4 py-1 my-4 bg-slate-800/30 italic text-slate-400" {...props} />,
                }}
              >
                {RUN_GUIDE_MD}
              </ReactMarkdown>
            </div>
          </div>
        );
      case Tab.REQUIREMENTS:
        return <CodeBlock code={REQUIREMENTS_TXT} filename="requirements.txt" />;
      case Tab.AGENT:
        return <CodeBlock code={getAgentCode()} filename={getAgentFilename()} />;
      case Tab.MAIN:
        return <CodeBlock code={MAIN_PY} filename="main.py" />;
      default:
        return null;
    }
  };

  return (
    <div className={`flex flex-col h-[100dvh] bg-slate-950 text-slate-200 overflow-hidden font-sans ${isResizing ? 'cursor-col-resize select-none' : ''}`}>
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
        
        <button 
          onClick={() => setShowChat(!showChat)}
          className="hidden md:block px-4 py-2 rounded-md text-sm font-medium transition-all bg-slate-800 text-slate-400 hover:text-white border border-slate-700"
        >
          {showChat ? 'Hide Assistant' : 'Show Assistant'}
        </button>

        <button 
          onClick={() => setShowChat(true)}
          className="md:hidden p-2 text-purple-400 hover:text-white"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" /></svg>
        </button>
      </header>

      {/* Main Layout */}
      <div className="flex-grow flex overflow-hidden relative">
        {/* Sidebar */}
        <nav className="hidden md:flex flex-none w-64 bg-slate-900 border-r border-slate-800 flex-col p-4 gap-2">
          
          {/* 레벨 선택기 */}
          <div className="mb-4 px-2">
            <label className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block mb-2">
              Select Level
            </label>
            <div className="relative">
              <select 
                value={currentLevel}
                onChange={(e) => setCurrentLevel(e.target.value as Level)}
                className="w-full bg-slate-800 text-white text-xs rounded-md border border-slate-600 p-2.5 pr-8 focus:ring-2 focus:ring-purple-500 outline-none appearance-none cursor-pointer hover:bg-slate-700 transition-colors"
              >
                {Object.values(Level).map(lvl => (
                  <option key={lvl} value={lvl}>{lvl}</option>
                ))}
              </select>
              <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-slate-400">
                <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z"/></svg>
              </div>
            </div>
          </div>
          
          <div className="h-px bg-slate-800 mb-2 mx-2"></div>

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
        <main className="flex-grow flex flex-col relative min-w-0">
          <div className="flex-grow overflow-hidden p-4 md:p-6">
             {renderContent()}
          </div>
        </main>
        
        {/* Resizer Handle */}
        {!isMobile && showChat && (
          <div
            className="w-1.5 bg-slate-900 hover:bg-purple-500 cursor-col-resize flex items-center justify-center transition-colors z-30"
            onMouseDown={startResizing}
          >
            <div className="w-0.5 h-8 bg-slate-600 rounded-full"></div>
          </div>
        )}

        {/* Chat Panel */}
        <aside 
          ref={sidebarRef}
          className={`hidden md:block bg-slate-900 border-l border-slate-800 transition-none ${!showChat && 'hidden'}`}
          style={{ width: showChat ? chatWidth : 0 }}
        >
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