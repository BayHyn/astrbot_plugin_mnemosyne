// Mnemosyne Memory Manager - 前端应用
class MnemosyneApp {
    constructor() {
        this.currentPage = 1;
        this.pageSize = 20;
        this.searchTerm = '';
        this.sessionFilter = '';
        this.deleteModal = null;
        this.timeChart = null;
        this.sessionChart = null;
        this.currentCollection = null;

        // 认证相关
        this.authEnabled = false;
        this.accessToken = null;
        this.isAuthenticated = false;
        this.loginModal = null;

        this.init();
    }
    
    async init() {
        // 初始化模态框
        this.deleteModal = new bootstrap.Modal(document.getElementById('deleteModal'));
        this.loginModal = new bootstrap.Modal(document.getElementById('loginModal'));

        // 设置登录事件
        this.setupLoginEvents();

        // 检查认证状态
        await this.checkAuth();

        // 如果需要认证但未认证，显示登录框
        if (this.authEnabled && !this.isAuthenticated) {
            this.showLoginModal();
            return;
        }

        // 加载初始数据
        await this.loadInitialData();

        // 更新认证状态显示
        this.updateAuthDisplay();

        // 设置所有事件监听器
        this.setupEventListeners();

        // 设置定时刷新
        setInterval(() => {
            if (this.isAuthenticated || !this.authEnabled) {
                this.loadStatus();
            }
        }, 30000); // 30秒刷新一次状态
    }

    setupLoginEvents() {
        // 登录按钮事件
        const loginSubmit = document.getElementById('login-submit');
        const tokenInput = document.getElementById('access-token-input');
        const logoutBtn = document.getElementById('logout-btn');

        if (loginSubmit) {
            loginSubmit.addEventListener('click', () => this.handleLogin());
        }

        if (tokenInput) {
            tokenInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.handleLogin();
                }
            });
        }

        if (logoutBtn) {
            logoutBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.handleLogout();
            });
        }
    }

    async loadInitialData() {
        await this.loadStatus();
        await this.loadStatistics();
        await this.loadMemories();
    }

    async checkAuth() {
        try {
            const response = await fetch('/api/auth/token');
            const result = await response.json();

            this.authEnabled = result.auth_enabled;

            if (!this.authEnabled) {
                // 认证已禁用，直接标记为已认证
                this.isAuthenticated = true;
                console.log('认证已禁用');
                return;
            }

            // 认证已启用，检查是否有存储的令牌
            const storedToken = localStorage.getItem('mnemosyne_access_token');
            if (storedToken) {
                this.accessToken = storedToken;
                // 验证存储的令牌是否有效
                const isValid = await this.validateToken(storedToken);
                if (isValid) {
                    this.isAuthenticated = true;
                    console.log('使用存储的访问令牌');
                    return;
                } else {
                    // 令牌无效，清除存储
                    localStorage.removeItem('mnemosyne_access_token');
                }
            }

            console.log('需要用户输入访问令牌');

        } catch (error) {
            console.error('检查认证状态失败:', error);
            this.showError('无法连接到服务器');
        }
    }

    async validateToken(token) {
        try {
            const response = await fetch('/api/status', {
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                }
            });
            return response.status === 200;
        } catch (error) {
            console.error('验证令牌失败:', error);
            return false;
        }
    }

    showLoginModal() {
        if (this.loginModal) {
            this.loginModal.show();
        }
    }

    hideLoginModal() {
        if (this.loginModal) {
            this.loginModal.hide();
        }
    }

    async handleLogin() {
        const tokenInput = document.getElementById('access-token-input');
        const loginError = document.getElementById('login-error');
        const loginErrorMessage = document.getElementById('login-error-message');
        const loginSubmit = document.getElementById('login-submit');

        if (!tokenInput) return;

        const token = tokenInput.value.trim();
        if (!token) {
            this.showLoginError('请输入访问令牌');
            return;
        }

        // 显示加载状态
        loginSubmit.disabled = true;
        loginSubmit.innerHTML = '<i class="bi bi-hourglass-split"></i> 验证中...';

        // 隐藏错误信息
        if (loginError) {
            loginError.style.display = 'none';
        }

        try {
            // 验证令牌
            const isValid = await this.validateToken(token);

            if (isValid) {
                // 令牌有效，保存并标记为已认证
                this.accessToken = token;
                this.isAuthenticated = true;
                localStorage.setItem('mnemosyne_access_token', token);

                // 隐藏登录框
                this.hideLoginModal();

                // 加载数据
                await this.loadInitialData();

                // 显示成功消息
                this.showSuccess('登录成功！');

                // 更新认证状态显示
                this.updateAuthDisplay();

            } else {
                this.showLoginError('访问令牌无效，请检查后重试');
            }

        } catch (error) {
            console.error('登录失败:', error);
            this.showLoginError('登录失败，请检查网络连接');
        } finally {
            // 恢复按钮状态
            loginSubmit.disabled = false;
            loginSubmit.innerHTML = '<i class="bi bi-check-circle"></i> 验证并登录';
        }
    }

    showLoginError(message) {
        const loginError = document.getElementById('login-error');
        const loginErrorMessage = document.getElementById('login-error-message');

        if (loginError && loginErrorMessage) {
            loginErrorMessage.textContent = message;
            loginError.style.display = 'block';
        }
    }

    handleLogout() {
        // 清除认证信息
        this.accessToken = null;
        this.isAuthenticated = false;
        localStorage.removeItem('mnemosyne_access_token');

        // 更新显示
        this.updateAuthDisplay();

        // 清空页面内容
        this.clearPageContent();

        // 显示登录框
        this.showLoginModal();

        this.showSuccess('已退出登录');
    }

    updateAuthDisplay() {
        const authDropdown = document.getElementById('auth-dropdown');

        if (authDropdown) {
            if (this.authEnabled && this.isAuthenticated) {
                authDropdown.style.display = 'block';
            } else {
                authDropdown.style.display = 'none';
            }
        }
    }

    clearPageContent() {
        // 清空统计数据
        document.getElementById('total-memories').textContent = '-';
        document.getElementById('unique-sessions').textContent = '-';
        document.getElementById('unique-personalities').textContent = '-';
        document.getElementById('db-type').textContent = '-';

        // 清空记忆列表
        const container = document.getElementById('memories-container');
        if (container) {
            container.innerHTML = `
                <div class="text-center py-4">
                    <i class="bi bi-lock" style="font-size: 3rem; color: #dee2e6;"></i>
                    <p class="text-muted mt-2">请先登录以查看数据</p>
                </div>
            `;
        }

        // 隐藏分页
        const paginationContainer = document.getElementById('pagination-container');
        if (paginationContainer) {
            paginationContainer.style.display = 'none';
        }

        // 清空图表
        if (this.timeChart) {
            this.timeChart.destroy();
            this.timeChart = null;
        }
        if (this.sessionChart) {
            this.sessionChart.destroy();
            this.sessionChart = null;
        }
    }

    getAuthHeaders() {
        const headers = {
            'Content-Type': 'application/json'
        };

        if (this.authEnabled && this.accessToken) {
            headers['Authorization'] = `Bearer ${this.accessToken}`;
        }

        return headers;
    }
    
    async loadStatus() {
        if (this.authEnabled && !this.isAuthenticated) {
            return;
        }

        try {
            const response = await fetch('/api/status', {
                headers: this.getAuthHeaders()
            });
            const result = await response.json();
            
            if (result.status === 'success') {
                this.updateStatusIndicator(true, result.data);
                
                // 设置当前集合
                if (!this.currentCollection && result.data.config) {
                    this.currentCollection = result.data.config.collection_name;
                }

            } else {
                this.updateStatusIndicator(false, null, result.message);
            }
        } catch (error) {
            console.error('加载状态失败:', error);
            this.updateStatusIndicator(false, null, '连接失败');
        }
    }
    
    updateStatusIndicator(connected, data, errorMessage) {
        const indicator = document.getElementById('status-indicator');
        
        if (connected && data) {
            indicator.className = 'status-badge bg-success';
            indicator.innerHTML = '<i class="bi bi-circle-fill"></i> 已连接';
            
            // 更新数据库类型
            const dbType = data.database?.info?.type || '未知';
            document.getElementById('db-type').textContent = dbType.toUpperCase();
        } else {
            indicator.className = 'status-badge bg-danger';
            indicator.innerHTML = '<i class="bi bi-circle-fill"></i> ' + (errorMessage || '连接失败');
            document.getElementById('db-type').textContent = '-';
        }
    }
    
    async loadStatistics() {
        if (this.authEnabled && !this.isAuthenticated) {
            return;
        }

        try {
            if (!this.currentCollection) return;

            const response = await fetch(`/api/statistics?collection_name=${this.currentCollection}`, {
                headers: this.getAuthHeaders()
            });
            const result = await response.json();
            
            if (result.status === 'success') {
                const stats = result.data;
                
                // 更新统计数字
                document.getElementById('total-memories').textContent = stats.total_memories || 0;
                document.getElementById('unique-sessions').textContent = stats.unique_sessions || 0;
                document.getElementById('unique-personalities').textContent = stats.unique_personalities || 0;
                
                // 更新图表
                this.updateTimeChart(stats.time_distribution || {});
                this.updateSessionChart(stats.session_distribution || {});
                
                // 更新会话过滤器
                this.updateSessionFilter(stats.session_distribution || {});
            }
        } catch (error) {
            console.error('加载统计信息失败:', error);
        }
    }
    
    updateTimeChart(timeData) {
        const ctx = document.getElementById('timeChart').getContext('2d');
        
        if (this.timeChart) {
            this.timeChart.destroy();
        }
        
        const dates = Object.keys(timeData).sort();
        const counts = dates.map(date => timeData[date]);
        
        this.timeChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [{
                    label: '记忆数量',
                    data: counts,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1
                        }
                    }
                }
            }
        });
    }
    
    updateSessionChart(sessionData) {
        const ctx = document.getElementById('sessionChart').getContext('2d');
        
        if (this.sessionChart) {
            this.sessionChart.destroy();
        }
        
        // 只显示前5个会话
        const sortedSessions = Object.entries(sessionData)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 5);
        
        const labels = sortedSessions.map(([session, ]) => 
            session.length > 10 ? session.substring(0, 10) + '...' : session
        );
        const data = sortedSessions.map(([, count]) => count);
        
        const colors = [
            '#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe'
        ];
        
        this.sessionChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: colors,
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 10,
                            usePointStyle: true
                        }
                    }
                }
            }
        });
    }
    
    updateSessionFilter(sessionData) {
        const select = document.getElementById('session-filter');
        
        // 清空现有选项（保留"所有会话"）
        while (select.children.length > 1) {
            select.removeChild(select.lastChild);
        }
        
        // 添加会话选项
        Object.keys(sessionData).forEach(sessionId => {
            const option = document.createElement('option');
            option.value = sessionId;
            option.textContent = `${sessionId} (${sessionData[sessionId]}条)`;
            select.appendChild(option);
        });
    }
    
    async loadMemories() {
        if (this.authEnabled && !this.isAuthenticated) {
            return;
        }

        const container = document.getElementById('memories-container');
        const loading = container.querySelector('.loading');

        // 显示加载状态
        loading.style.display = 'block';
        
        try {
            const params = new URLSearchParams({
                page: this.currentPage,
                page_size: this.pageSize
            });
            
            if (this.searchTerm) {
                params.append('search', this.searchTerm);
            }
            
            if (this.sessionFilter) {
                params.append('session_id', this.sessionFilter);
            }
            
            if (!this.currentCollection) {
                this.showError('请先选择一个集合');
                return;
            }
            params.append('collection_name', this.currentCollection);

            const response = await fetch(`/api/memories?${params}`, {
                headers: this.getAuthHeaders()
            });
            const result = await response.json();
            
            if (result.status === 'success') {
                this.renderMemories(result.data.records);
                this.renderPagination(result.data.pagination);
            } else {
                this.showError('加载记忆数据失败: ' + result.message);
            }
        } catch (error) {
            console.error('加载记忆数据失败:', error);
            this.showError('加载记忆数据失败: ' + error.message);
        } finally {
            loading.style.display = 'none';
        }
    }
    
    renderMemories(memories) {
        const container = document.getElementById('memories-container');
        
        if (memories.length === 0) {
            container.innerHTML = `
                <div class="text-center py-4">
                    <i class="bi bi-inbox" style="font-size: 3rem; color: #dee2e6;"></i>
                    <p class="text-muted mt-2">暂无记忆数据</p>
                </div>
            `;
            return;
        }
        
        const memoriesHtml = memories.map(memory => `
            <div class="memory-item">
                <div class="d-flex justify-content-between align-items-start">
                    <div class="flex-grow-1">
                        <div class="memory-content">${this.escapeHtml(memory.content)}</div>
                        <div class="memory-meta">
                            <span><i class="bi bi-person"></i> 会话: ${this.escapeHtml(memory.session_id)}</span>
                            <span class="ms-3"><i class="bi bi-clock"></i> ${memory.create_time_str}</span>
                            ${memory.personality_id ? `<span class="ms-3"><i class="bi bi-mask"></i> ${this.escapeHtml(memory.personality_id)}</span>` : ''}
                        </div>
                    </div>
                    <div class="ms-3">
                        <button class="btn btn-outline-danger btn-sm" onclick="app.confirmDelete({id: ${memory.memory_id}, type: 'memory', collection: '${this.currentCollection}'})">
                            <i class="bi bi-trash"></i>
                        </button>
                    </div>
                </div>
            </div>
        `).join('');
        
        container.innerHTML = memoriesHtml;
    }
    
    renderPagination(pagination) {
        const container = document.getElementById('pagination-container');
        const paginationEl = document.getElementById('pagination');
        
        if (pagination.total_pages <= 1) {
            container.style.display = 'none';
            return;
        }
        
        container.style.display = 'block';
        
        let paginationHtml = '';
        
        // 上一页
        if (pagination.page > 1) {
            paginationHtml += `
                <li class="page-item">
                    <a class="page-link" href="#" onclick="app.goToPage(${pagination.page - 1})">
                        <i class="bi bi-chevron-left"></i>
                    </a>
                </li>
            `;
        }
        
        // 页码
        const startPage = Math.max(1, pagination.page - 2);
        const endPage = Math.min(pagination.total_pages, pagination.page + 2);
        
        for (let i = startPage; i <= endPage; i++) {
            paginationHtml += `
                <li class="page-item ${i === pagination.page ? 'active' : ''}">
                    <a class="page-link" href="#" onclick="app.goToPage(${i})">${i}</a>
                </li>
            `;
        }
        
        // 下一页
        if (pagination.page < pagination.total_pages) {
            paginationHtml += `
                <li class="page-item">
                    <a class="page-link" href="#" onclick="app.goToPage(${pagination.page + 1})">
                        <i class="bi bi-chevron-right"></i>
                    </a>
                </li>
            `;
        }
        
        paginationEl.innerHTML = paginationHtml;
    }
    
    goToPage(page) {
        this.currentPage = page;
        this.loadMemories();
    }
    
    confirmDelete(options) {
        const { id, type, collection } = options;
        const modal = document.getElementById('deleteModal');
        const message = document.getElementById('delete-message');
        const confirmBtn = document.getElementById('confirm-delete');
        
        if (type === 'memory') {
            message.textContent = `确定要从集合 [${collection}] 中删除这条记忆吗？此操作不可撤销。`;
            confirmBtn.onclick = () => this.deleteMemory(collection, id);
        } else if (type === 'session') {
            message.textContent = `确定要从集合 [${collection}] 中删除会话 [${id}] 的所有记忆吗？此操作不可撤销。`;
            confirmBtn.onclick = () => this.deleteSessionMemories(collection, id);
        }
        
        this.deleteModal.show();
    }
    
    async deleteMemory(collectionName, memoryId) {
        try {
            const response = await fetch(`/api/collections/${collectionName}/memories/${memoryId}`, {
                method: 'DELETE',
                headers: this.getAuthHeaders()
            });
            const result = await response.json();
            
            if (result.status === 'success') {
                this.showSuccess('记忆删除成功');
                this.loadMemories();
                this.loadStatistics();
            } else {
                this.showError('删除失败: ' + result.message);
            }
        } catch (error) {
            console.error('删除记忆失败:', error);
            this.showError('删除失败: ' + error.message);
        } finally {
            this.deleteModal.hide();
        }
    }
    
    async deleteSessionMemories(collectionName, sessionId) {
        try {
            const response = await fetch(`/api/collections/${collectionName}/sessions/${encodeURIComponent(sessionId)}`, {
                method: 'DELETE',
                headers: this.getAuthHeaders()
            });
            const result = await response.json();
            
            if (result.status === 'success') {
                this.showSuccess('会话记忆删除成功');
                this.loadMemories();
                this.loadStatistics();
            } else {
                this.showError('删除失败: ' + result.message);
            }
        } catch (error) {
            console.error('删除会话记忆失败:', error);
            this.showError('删除失败: ' + error.message);
        } finally {
            this.deleteModal.hide();
        }
    }
    
    showSuccess(message) {
        this.showAlert(message, 'success');
    }
    
    showError(message) {
        this.showAlert(message, 'danger');
    }
    
    showAlert(message, type) {
        const alertHtml = `
            <div class="alert alert-${type} alert-custom alert-dismissible fade show" role="alert">
                <i class="bi bi-${type === 'success' ? 'check-circle' : 'exclamation-triangle'}"></i>
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        
        const container = document.querySelector('.container');
        container.insertAdjacentHTML('afterbegin', alertHtml);
        
        // 3秒后自动消失
        setTimeout(() => {
            const alert = container.querySelector('.alert');
            if (alert) {
                alert.remove();
            }
        }, 3000);
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    async refreshData() {
        if (!this.currentCollection) {
            await this.loadStatus();
        }
        if (!this.currentCollection) {
            this.showError("无法确定当前集合，请刷新页面。");
            return;
        }
        await this.loadStatistics();
        await this.loadMemories();
    }

    setupEventListeners() {
        // 搜索
        const searchInput = document.getElementById('search-input');
        const searchButton = document.getElementById('search-button');
        if (searchButton) {
            searchButton.addEventListener('click', () => this.handleSearch());
        }
        if (searchInput) {
            searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') this.handleSearch();
            });
        }

        // 过滤器
        const sessionFilter = document.getElementById('session-filter');
        if (sessionFilter) {
            sessionFilter.addEventListener('change', () => this.handleSessionFilter());
        }

        // 分页大小
        const pageSizeSelect = document.getElementById('page-size');
        if (pageSizeSelect) {
            pageSizeSelect.addEventListener('change', () => this.handlePageSizeChange());
        }

        // 刷新按钮
        const refreshButton = document.getElementById('refresh-button');
        if (refreshButton) {
            refreshButton.addEventListener('click', () => this.refreshData());
        }
    }

    handleSearch() {
        const searchInput = document.getElementById('search-input');
        this.searchTerm = searchInput.value.trim();
        this.currentPage = 1;
        this.loadMemories();
    }

    handleSessionFilter() {
        const sessionFilter = document.getElementById('session-filter');
        this.sessionFilter = sessionFilter.value;
        this.currentPage = 1;
        this.loadMemories();
    }

    handlePageSizeChange() {
        const pageSizeSelect = document.getElementById('page-size');
        this.pageSize = parseInt(pageSizeSelect.value);
        this.currentPage = 1;
        this.loadMemories();
    }
}

// 初始化应用
let app;
document.addEventListener('DOMContentLoaded', function() {
    app = new MnemosyneApp();
});
