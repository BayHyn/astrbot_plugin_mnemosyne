// Mnemosyne 管理面板主脚本

// API 基础配置
const API_BASE = '/api';

// 全局状态
const AppState = {
    currentPage: 'dashboard',
    loading: false,
    dashboardData: null,
    memoriesData: null,
    sessionsData: null,
    apiKey: null,  // API 密钥
};

// ==================== API Key 管理 ====================

/**
 * 从 localStorage 加载 API Key
 */
function loadApiKey() {
    const savedKey = localStorage.getItem('mnemosyne_api_key');
    if (savedKey) {
        AppState.apiKey = savedKey;
        return true;
    }
    return false;
}

/**
 * 保存 API Key 到 localStorage
 * @param {string} apiKey - API 密钥
 */
function saveApiKey(apiKey) {
    if (apiKey && apiKey.trim()) {
        AppState.apiKey = apiKey.trim();
        localStorage.setItem('mnemosyne_api_key', AppState.apiKey);
        return true;
    }
    return false;
}

/**
 * 清除 API Key
 */
function clearApiKey() {
    AppState.apiKey = null;
    localStorage.removeItem('mnemosyne_api_key');
}

/**
 * 显示 API Key 输入对话框
 */
function promptApiKey() {
    const apiKey = prompt(
        '请输入管理面板 API 密钥：\n\n' +
        '提示：如果您未配置自定义密钥，请查看服务器日志获取自动生成的 token。\n' +
        '日志中会显示类似 "🔒 已生成动态强 token" 的信息。'
    );
    
    if (apiKey && apiKey.trim()) {
        if (saveApiKey(apiKey)) {
            showToast('API 密钥已保存', 'success');
            return true;
        }
    }
    return false;
}

/**
 * 检查是否需要输入 API Key
 */
function checkApiKey() {
    if (!AppState.apiKey) {
        if (!loadApiKey()) {
            return promptApiKey();
        }
    }
    return true;
}

// ==================== 安全函数 ====================

/**
 * HTML 转义函数 - 防止 XSS 攻击
 * 将特殊字符转换为 HTML 实体
 *
 * @param {string} unsafe - 未转义的字符串
 * @returns {string} - 转义后的安全字符串
 */
function escapeHtml(unsafe) {
    if (typeof unsafe !== 'string') {
        return String(unsafe);
    }
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

/**
 * 格式化时间 - 安全版本
 * @param {string|number} timestamp - 时间戳或日期字符串
 * @returns {string} - 格式化后的时间字符串
 */
function formatTime(timestamp) {
    if (!timestamp) return '-';
    
    try {
        const date = new Date(timestamp);
        if (isNaN(date.getTime())) {
            return escapeHtml(String(timestamp));
        }
        return date.toLocaleString('zh-CN', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    } catch (error) {
        return escapeHtml(String(timestamp));
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    console.log('Mnemosyne 管理面板初始化...');
    
    // 加载 API Key
    if (!checkApiKey()) {
        showToast('未设置 API 密钥，无法访问管理面板', 'error');
        return;
    }
    
    // 设置导航
    setupNavigation();
    
    // 加载初始页面
    loadPage('dashboard');
});

// 导航设置
function setupNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const page = item.dataset.page;
            
            // 更新导航状态
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');
            
            // 加载页面
            loadPage(page);
        });
    });
}

// 加载页面
function loadPage(pageName) {
    console.log(`加载页面: ${pageName}`);
    AppState.currentPage = pageName;
    
    // 隐藏所有页面
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    
    // 显示目标页面
    const targetPage = document.getElementById(`${pageName}-page`);
    if (targetPage) {
        targetPage.classList.add('active');
    }
    
    // 加载页面数据
    switch(pageName) {
        case 'dashboard':
            loadDashboard();
            break;
        case 'memories':
            // 记忆页面按需加载
            break;
        case 'sessions':
            loadSessions();
            break;
        case 'statistics':
            loadStatistics();
            break;
        case 'logs':
            loadLogs();
            break;
        case 'config':
            loadConfig();
            break;
    }
}

// API 调用封装
async function apiCall(endpoint, method = 'GET', body = null) {
    // 确保有 API Key
    if (!AppState.apiKey) {
        if (!checkApiKey()) {
            throw new Error('未设置 API 密钥');
        }
    }
    
    const options = {
        method: method,
        headers: {
            'Content-Type': 'application/json',
            'X-API-Key': AppState.apiKey,  // 添加认证 header
        },
    };
    
    if (body && (method === 'POST' || method === 'PUT' || method === 'PATCH')) {
        options.body = JSON.stringify(body);
    }
    
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, options);
        const data = await response.json();
        
        // 处理认证失败
        if (response.status === 401 || data.error === 'Unauthorized') {
            showToast('认证失败，请重新输入 API 密钥', 'error');
            clearApiKey();
            if (promptApiKey()) {
                // 重试请求
                return apiCall(endpoint, method, body);
            }
            throw new Error('认证失败');
        }
        
        if (!data.success) {
            throw new Error(data.error || '请求失败');
        }
        
        return data.data;
    } catch (error) {
        console.error(`API 调用失败 [${endpoint}]:`, error);
        
        // 网络错误或其他错误
        if (error.message !== '认证失败') {
            showToast(`请求失败: ${error.message}`, 'error');
        }
        throw error;
    }
}

// 显示加载状态
function showLoading(show = true) {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.style.display = show ? 'flex' : 'none';
    }
    AppState.loading = show;
}

// Toast 通知
function showToast(message, type = 'success') {
    const container = document.getElementById('toast-container');
    if (!container) return;
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    
    container.appendChild(toast);
    
    // 3秒后自动移除
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => {
            container.removeChild(toast);
        }, 300);
    }, 3000);
}

// 格式化日期时间（向后兼容）
function formatDateTime(dateTimeStr) {
    return formatTime(dateTimeStr);
}

// 格式化数字
function formatNumber(num) {
    if (typeof num !== 'number') return '-';
    
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

// 格式化字节大小
function formatBytes(bytes) {
    if (typeof bytes !== 'number') return '-';
    
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let size = bytes;
    let unitIndex = 0;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
        size /= 1024;
        unitIndex++;
    }
    
    return `${size.toFixed(2)} ${units[unitIndex]}`;
}

// 获取状态颜色和图标
function getStatusIndicator(status) {
    const indicators = {
        'healthy': { icon: '🟢', text: '健康', class: 'healthy' },
        'unhealthy': { icon: '🔴', text: '异常', class: 'unhealthy' },
        'degraded': { icon: '🟡', text: '降级', class: 'degraded' },
        'unknown': { icon: '⚪', text: '未知', class: 'unknown' }
    };
    
    return indicators[status] || indicators['unknown'];
}

// 日志查看功能（占位）
function loadLogs() {
    console.log('日志查看功能待实现');
    showToast('日志查看功能正在开发中', 'warning');
}

// 配置管理功能（占位）
function loadConfig() {
    console.log('配置管理功能待实现');
    showToast('配置管理功能正在开发中', 'warning');
}

function saveConfig() {
    showToast('配置保存功能正在开发中', 'warning');
}

// 导出工具函数
window.AppState = AppState;
window.apiCall = apiCall;
window.showLoading = showLoading;
window.showToast = showToast;
window.escapeHtml = escapeHtml;  // 导出 XSS 防护函数
window.formatTime = formatTime;  // 导出安全的时间格式化函数
window.formatDateTime = formatDateTime;
window.formatNumber = formatNumber;
window.formatBytes = formatBytes;
window.getStatusIndicator = getStatusIndicator;
window.loadPage = loadPage;
window.loadLogs = loadLogs;
window.loadConfig = loadConfig;
window.saveConfig = saveConfig;