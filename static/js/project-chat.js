// project-chat.js - Sistema de Chat Inteligente com Analista de Requisitos

class ProjectChatSystem {
    constructor() {
        this.selectedProjectType = null;
        this.uploadedFiles = [];
        this.chatHistory = [];
        this.teamMembers = {};
        this.isAnalystActive = false;
        this.securityFilters = this.initSecurityFilters();
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupFileUpload();
        this.autoResizeTextarea();
    }

    initSecurityFilters() {
        return {
            // Palavras/frases que indicam informações sensíveis
            sensitiveKeywords: [
                'senha', 'password', 'token', 'api key', 'chave', 'secret',
                'servidor interno', 'ip interno', 'credencial', 'authentication',
                'infraestrutura', 'arquitetura interna', 'código fonte',
                'vulnerabilidade', 'exploit', 'backdoor', 'configuração do servidor',
                'base de dados', 'database password', 'connection string'
            ],
            
            // Padrões regex para detectar informações técnicas sensíveis
            sensitivePatterns: [
                /\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/g, // IPs
                /[a-zA-Z0-9]{20,}/g, // Possíveis tokens/chaves
                /postgres:\/\/|mysql:\/\/|mongodb:\/\//g, // Connection strings
                /Bearer\s+[a-zA-Z0-9\-\._~\+\/]+=*/g // Bearer tokens
            ]
        };
    }

    setupEventListeners() {
        // Project type selection
        document.querySelectorAll('.project-type-card').forEach(card => {
            card.addEventListener('click', () => this.selectProjectType(card));
        });

        // Start project button
        document.getElementById('startProjectBtn').addEventListener('click', () => {
            this.startProject();
        });

        // Chat input
        const chatInput = document.getElementById('chatInput');
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Send button
        document.getElementById('sendBtn').addEventListener('click', () => {
            this.sendMessage();
        });

        // Upload button
        document.getElementById('uploadBtn').addEventListener('click', () => {
            this.toggleFileUpload();
        });
    }

    setupFileUpload() {
        const fileUploadArea = document.getElementById('fileUploadArea');
        const fileInput = document.getElementById('fileInput');

        // Drag and drop
        fileUploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            fileUploadArea.classList.add('dragover');
        });

        fileUploadArea.addEventListener('dragleave', () => {
            fileUploadArea.classList.remove('dragover');
        });

        fileUploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            fileUploadArea.classList.remove('dragover');
            this.handleFiles(e.dataTransfer.files);
        });

        // Click to upload
        fileUploadArea.addEventListener('click', () => {
            fileInput.click();
        });

        fileInput.addEventListener('change', (e) => {
            this.handleFiles(e.target.files);
        });
    }

    autoResizeTextarea() {
        const textarea = document.getElementById('chatInput');
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });
    }

    selectProjectType(card) {
        // Remove previous selection
        document.querySelectorAll('.project-type-card').forEach(c => {
            c.classList.remove('selected');
        });

        // Add selection to clicked card
        card.classList.add('selected');
        this.selectedProjectType = card.dataset.type;

        // Enable start button
        document.getElementById('startProjectBtn').disabled = false;
    }

    startProject() {
        if (!this.selectedProjectType) return;

        // Hide project type section
        document.getElementById('projectTypeSection').style.display = 'none';
        
        // Show chat section
        document.getElementById('chatSection').style.display = 'block';

        // Update project type in header
        document.getElementById('selectedProjectType').textContent = 
            this.getProjectTypeName(this.selectedProjectType);

        // Setup team members
        this.setupTeamMembers();

        // Start conversation with analyst
        this.startInitialConversation();
    }

    getProjectTypeName(type) {
        const names = {
            'development': 'Desenvolvimento',
            'automation': 'Automação',
            'analytics': 'Analytics & IA',
            'integration': 'Integração',
            'consulting': 'Consultoria',
            'custom': 'Personalizado'
        };
        return names[type] || 'Projeto';
    }

    setupTeamMembers() {
        const teamMembersContainer = document.getElementById('teamMembers');
        
        // Define team based on project type
        const teams = {
            development: [
                { name: 'Ana Silva', role: 'Analista de Requisitos', avatar: 'AR', status: 'online' },
                { name: 'Carlos Dev', role: 'Tech Lead', avatar: 'CD', status: 'online' },
                { name: 'Julia UX', role: 'UX Designer', avatar: 'JU', status: 'online' },
                { name: 'Pedro Full', role: 'Full-Stack Dev', avatar: 'PF', status: 'online' }
            ],
            automation: [
                { name: 'Ana Silva', role: 'Analista de Requisitos', avatar: 'AR', status: 'online' },
                { name: 'Roberto RPA', role: 'RPA Specialist', avatar: 'RR', status: 'online' },
                { name: 'Marina Auto', role: 'Automation Engineer', avatar: 'MA', status: 'online' }
            ],
            analytics: [
                { name: 'Ana Silva', role: 'Analista de Requisitos', avatar: 'AR', status: 'online' },
                { name: 'Dr. Data', role: 'Data Scientist', avatar: 'DD', status: 'online' },
                { name: 'Luis ML', role: 'ML Engineer', avatar: 'LM', status: 'online' },
                { name: 'Sara Viz', role: 'Data Visualization', avatar: 'SV', status: 'online' }
            ],
            integration: [
                { name: 'Ana Silva', role: 'Analista de Requisitos', avatar: 'AR', status: 'online' },
                { name: 'API Master', role: 'Integration Architect', avatar: 'AM', status: 'online' },
                { name: 'Micro Mike', role: 'Microservices Dev', avatar: 'MM', status: 'online' }
            ],
            consulting: [
                { name: 'Ana Silva', role: 'Analista de Requisitos', avatar: 'AR', status: 'online' },
                { name: 'Strategy Sam', role: 'Strategy Consultant', avatar: 'SS', status: 'online' },
                { name: 'Arch Anna', role: 'Solution Architect', avatar: 'AA', status: 'online' }
            ],
            custom: [
                { name: 'Ana Silva', role: 'Analista de Requisitos', avatar: 'AR', status: 'online' },
                { name: 'Flex Team', role: 'Adaptive Specialists', avatar: 'FT', status: 'online' }
            ]
        };

        const team = teams[this.selectedProjectType] || teams.custom;
        this.teamMembers = team;

        teamMembersContainer.innerHTML = team.map(member => `
            <div class="team-member">
                <div class="member-avatar">${member.avatar}</div>
                <div class="member-info">
                    <div style="font-weight: 500;">${member.name}</div>
                    <div style="font-size: 0.75rem; opacity: 0.7;">${member.role}</div>
                </div>
                <div class="member-status"></div>
            </div>
        `).join('');
    }

    startInitialConversation() {
        // Ana Silva (Analista de Requisitos) sempre inicia a conversa
        setTimeout(() => {
            this.addMessage('analyst', 'Ana Silva', 
                'Olá! Sou a Ana Silva, Analista de Requisitos Senior da equipe. Vou ajudar você a definir todos os detalhes do seu projeto para garantir que entregamos exatamente o que você precisa.');
        }, 1000);

        setTimeout(() => {
            this.addMessage('analyst', 'Ana Silva', 
                `Para começar, vou fazer algumas perguntas sobre seu projeto de ${this.getProjectTypeName(this.selectedProjectType).toLowerCase()}. Pode me contar qual é o objetivo principal e o que você gostaria de alcançar?`);
        }, 2500);

        this.isAnalystActive = true;
    }

    addMessage(type, sender, text, timestamp = null) {
        const messagesContainer = document.getElementById('chatMessages');
        const messageTime = timestamp || new Date().toLocaleTimeString('pt-BR', { 
            hour: '2-digit', 
            minute: '2-digit' 
        });

        const messageElement = document.createElement('div');
        messageElement.className = `message ${type}`;
        
        messageElement.innerHTML = `
            <div class="message-avatar">${this.getAvatarText(sender, type)}</div>
            <div class="message-content">
                <div class="message-sender">${sender}</div>
                <div class="message-text">${text}</div>
                <div class="message-timestamp">${messageTime}</div>
            </div>
        `;

        messagesContainer.appendChild(messageElement);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        // Add to chat history
        this.chatHistory.push({
            type, sender, text, timestamp: messageTime
        });
    }

    getAvatarText(sender, type) {
        if (type === 'user') return '<i class="fas fa-user"></i>';
        
        // Get avatar from team members
        const member = this.teamMembers.find(m => m.name === sender);
        return member ? member.avatar : '<i class="fas fa-robot"></i>';
    }

    sendMessage() {
        const input = document.getElementById('chatInput');
        const message = input.value.trim();
        
        if (!message && this.uploadedFiles.length === 0) return;

        // Security check
        if (this.containsSensitiveInfo(message)) {
            this.showSecurityWarning();
            return;
        }

        // Add user message
        let userMessage = message;
        if (this.uploadedFiles.length > 0) {
            const fileList = this.uploadedFiles.map(f => f.name).join(', ');
            userMessage += `\n\n📎 Arquivos anexados: ${fileList}`;
        }

        this.addMessage('user', 'Você', userMessage);

        // Clear input and files
        input.value = '';
        input.style.height = 'auto';
        this.clearUploadedFiles();

        // Generate response
        this.generateResponse(message);
    }

    containsSensitiveInfo(text) {
        const lowerText = text.toLowerCase();
        
        // Check sensitive keywords
        for (const keyword of this.securityFilters.sensitiveKeywords) {
            if (lowerText.includes(keyword.toLowerCase())) {
                return true;
            }
        }

        // Check sensitive patterns
        for (const pattern of this.securityFilters.sensitivePatterns) {
            if (pattern.test(text)) {
                return true;
            }
        }

        return false;
    }

    showSecurityWarning() {
        this.addMessage('analyst', 'Ana Silva', 
            '⚠️ Por questões de segurança, evite compartilhar informações sensíveis como senhas, tokens ou detalhes técnicos internos. Vou solicitar apenas as informações necessárias para o projeto de forma segura.');
    }

    async generateResponse(userMessage) {
        // Show typing indicator
        this.showTypingIndicator();

        try {
            // Call real API
            const response = await fetch('/api/project/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: userMessage,
                    project_type: this.selectedProjectType,
                    files: this.uploadedFiles.map(f => ({ name: f.name, size: f.size, type: f.type }))
                })
            });

            const data = await response.json();
            
            this.hideTypingIndicator();
            
            if (data.success && data.response) {
                this.addMessage('analyst', data.response.sender, data.response.text);
                
                // Se requer documentos, sugerir upload
                if (data.response.requires_documents) {
                    setTimeout(() => {
                        this.addMessage('analyst', 'Ana Silva', 
                            '📎 Para uma análise mais precisa, seria muito útil se você pudesse anexar alguns exemplos dos documentos que precisam ser processados. Use o botão de anexo abaixo.');
                    }, 1500);
                }
            } else {
                this.addMessage('analyst', 'Ana Silva', 'Desculpe, houve um problema. Pode repetir sua mensagem?');
            }

            // Às vezes outros membros da equipe também respondem
            if (Math.random() > 0.7) {
                setTimeout(() => {
                    this.generateTeamResponse(userMessage);
                }, 2000 + Math.random() * 3000);
            }

        } catch (error) {
            this.hideTypingIndicator();
            this.addMessage('analyst', 'Ana Silva', 'Houve um problema de conexão. Por favor, tente novamente.');
            console.error('Chat API error:', error);
        }
    }

    generateAnalystResponse(userMessage) {
        const lowerMessage = userMessage.toLowerCase();
        
        // Respostas contextuais da Analista de Requisitos
        if (lowerMessage.includes('automação') && lowerMessage.includes('documento')) {
            return 'Entendi que você precisa automatizar documentos. Para te ajudar melhor, você poderia enviar alguns exemplos dos documentos que precisam ser processados? Isso me ajudará a entender o formato e os tipos de dados envolvidos. Que tipo de automação você está pensando - extração de dados, geração automática, ou processamento/análise?';
        }
        
        if (lowerMessage.includes('dashboard') || lowerMessage.includes('relatório')) {
            return 'Perfeito! Para criar o dashboard ideal, preciso entender: Quais são os principais KPIs que você gostaria de visualizar? Com que frequência os dados precisam ser atualizados? E quem são os usuários finais que irão utilizar este dashboard?';
        }
        
        if (lowerMessage.includes('sistema') || lowerMessage.includes('aplicação')) {
            return 'Ótimo! Para desenvolver o sistema adequado às suas necessidades: Quantos usuários aproximadamente irão utilizar? O sistema precisa integrar com alguma ferramenta existente? E qual é o prazo ideal para a primeira versão?';
        }
        
        if (lowerMessage.includes('integração') || lowerMessage.includes('api')) {
            return 'Entendi que você precisa de integração entre sistemas. Quais são os sistemas/plataformas que precisam ser conectados? Que tipo de dados serão sincronizados? E com que frequência essa sincronização precisa acontecer?';
        }
        
        if (lowerMessage.includes('dados') || lowerMessage.includes('análise')) {
            return 'Excelente! Para a análise de dados ser efetiva: De onde vêm os dados (banco de dados, arquivos, APIs)? Que tipos de insights você espera descobrir? E como os resultados serão utilizados na tomada de decisão?';
        }

        // Respostas gerais de acompanhamento
        const generalResponses = [
            'Interessante! Para garantir que desenvolvemos a solução ideal, você poderia detalhar um pouco mais sobre o contexto atual e os principais desafios que está enfrentando?',
            'Perfeito! Estou mapeando seus requisitos. Para complementar, como você enxerga o sucesso deste projeto? Quais seriam os principais resultados esperados?',
            'Ótimo direcionamento! Para refinar nossa proposta, qual é o público-alvo desta solução? E existem restrições técnicas ou de orçamento que devo considerar?',
            'Entendi sua necessidade. Para estruturar melhor o projeto, você tem alguma referência ou exemplo de solução similar que considera ideal?'
        ];
        
        return generalResponses[Math.floor(Math.random() * generalResponses.length)];
    }

    generateTeamResponse(userMessage) {
        const availableMembers = this.teamMembers.filter(m => m.role !== 'Analista de Requisitos');
        if (availableMembers.length === 0) return;

        const member = availableMembers[Math.floor(Math.random() * availableMembers.length)];
        
        const teamResponses = [
            'Já estou pensando em algumas abordagens técnicas interessantes para isso!',
            'Vou começar a estruturar a arquitetura baseada no que você descreveu.',
            'Temos experiência com projetos similares, vai ficar excelente!',
            'Interessante! Vou preparar algumas opções de implementação.',
            'Perfeito! Já vislumbro como podemos otimizar isso.'
        ];

        const response = teamResponses[Math.floor(Math.random() * teamResponses.length)];
        this.addMessage('agent', member.name, response);
    }

    showTypingIndicator() {
        const messagesContainer = document.getElementById('chatMessages');
        const typingElement = document.createElement('div');
        typingElement.id = 'typingIndicator';
        typingElement.className = 'typing-indicator';
        
        typingElement.innerHTML = `
            <div class="member-avatar">AR</div>
            <div style="flex: 1;">
                Ana Silva está digitando
                <div class="typing-dots">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;

        messagesContainer.appendChild(typingElement);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    toggleFileUpload() {
        const uploadArea = document.getElementById('fileUploadArea');
        const isVisible = uploadArea.style.display !== 'none';
        uploadArea.style.display = isVisible ? 'none' : 'block';

        // Update button style
        const uploadBtn = document.getElementById('uploadBtn');
        if (isVisible) {
            uploadBtn.style.borderColor = 'rgba(255, 255, 255, 0.2)';
            uploadBtn.style.color = 'var(--text-light)';
        } else {
            uploadBtn.style.borderColor = 'var(--accent-green)';
            uploadBtn.style.color = 'var(--accent-green)';
        }
    }

    async handleFiles(files) {
        const formData = new FormData();
        const validFiles = [];

        for (let file of files) {
            if (file.size > 10 * 1024 * 1024) { // 10MB limit
                alert(`Arquivo ${file.name} é muito grande. Limite: 10MB`);
                continue;
            }

            formData.append('files', file);
            validFiles.push({
                name: file.name,
                size: file.size,
                type: file.type,
                file: file
            });
        }

        if (validFiles.length === 0) return;

        // Show upload progress
        this.addMessage('analyst', 'Sistema', 
            `📤 Processando ${validFiles.length} arquivo(s)...`);

        try {
            const response = await fetch('/api/project/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                // Add files to uploaded list
                this.uploadedFiles.push(...validFiles);
                this.updateUploadedFilesDisplay();

                // Show success message
                this.addMessage('analyst', 'Ana Silva', 
                    `✅ ${data.message}! Agora posso analisar o conteúdo e entender melhor suas necessidades. Com base nos documentos, posso fazer recomendações mais precisas.`);
            } else {
                this.addMessage('analyst', 'Sistema', 
                    `❌ Erro no upload: ${data.error}`);
            }
        } catch (error) {
            this.addMessage('analyst', 'Sistema', 
                '❌ Erro na conexão durante o upload. Tente novamente.');
            console.error('Upload error:', error);
        }
        
        // Hide upload area after processing
        document.getElementById('fileUploadArea').style.display = 'none';
        document.getElementById('uploadBtn').style.borderColor = 'rgba(255, 255, 255, 0.2)';
        document.getElementById('uploadBtn').style.color = 'var(--text-light)';
    }

    updateUploadedFilesDisplay() {
        const container = document.getElementById('uploadedFiles');
        container.innerHTML = this.uploadedFiles.map((file, index) => `
            <div class="uploaded-file">
                <i class="fas fa-file me-2"></i>
                ${file.name}
                <button class="file-remove" onclick="projectChat.removeFile(${index})">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `).join('');
    }

    removeFile(index) {
        this.uploadedFiles.splice(index, 1);
        this.updateUploadedFilesDisplay();
    }

    clearUploadedFiles() {
        this.uploadedFiles = [];
        this.updateUploadedFilesDisplay();
    }
}

// Initialize the system
const projectChat = new ProjectChatSystem();

// Make it globally available for inline event handlers
window.projectChat = projectChat;