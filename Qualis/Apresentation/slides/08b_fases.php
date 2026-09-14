<section class="slide" id="slide-8b">
    <div class="full-center fade-up" style="padding-top: 40px;">
        <h1 class="title-main" style="margin-bottom: 5px;">A Construção da Plataforma</h1>
        <p class="subtitle" style="margin-bottom: 50px;">O processo prático dividido em quatro fases de engenharia.</p>

        <div class="phases-container">

            <div class="phase-card">
                <div class="phase-icon"><i class="fas fa-cube"></i></div>
                <div class="phase-number">Fase 1</div>
                <h4>Modelagem</h4>
                <p>Definição de requisitos, emulação de sensores e criação do ambiente 3D na <strong>Unity</strong>.</p>
            </div>

            <div class="phase-arrow"><i class="fas fa-chevron-right"></i></div>

            <div class="phase-card">
                <div class="phase-icon blue"><i class="fas fa-server"></i></div>
                <div class="phase-number">Fase 2</div>
                <h4 style="color: var(--neon-blue);">Backend</h4>
                <p>Design dos endpoints RESTful em <strong>Node.js</strong> e esquema de coleções no <strong>MongoDB</strong>.</p>
            </div>

            <div class="phase-arrow"><i class="fas fa-chevron-right"></i></div>

            <div class="phase-card">
                <div class="phase-icon green"><i class="fas fa-cloud-upload-alt"></i></div>
                <div class="phase-number">Fase 3</div>
                <h4 style="color: var(--neon-green);">Implantação</h4>
                <p>Deploy na <strong>VPS Contabo (AlmaLinux)</strong> para simular ambiente remoto e tráfego de internet.</p>
            </div>

            <div class="phase-arrow"><i class="fas fa-chevron-right"></i></div>

            <div class="phase-card">
                <div class="phase-icon"><i class="fas fa-check-double"></i></div>
                <div class="phase-number">Fase 4</div>
                <h4>Validação</h4>
                <p>Testes <strong>ponta-a-ponta</strong>: do evento na Unity até a persistência JSON e interface web.</p>
            </div>

        </div>
    </div>
</section>