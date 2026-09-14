document.addEventListener('DOMContentLoaded', () => {
    const timelineItems = document.querySelectorAll('.timeline-nav-item');
    const modalTimeline = document.getElementById('modal-timeline');
    const closeTimeline = document.getElementById('btn-close-timeline');
    const detailsContent = document.getElementById('timeline-details-content');

    const timelineData = {
        "1": {
            title: "Ano 1 – Fase das Disciplinas",
            activities: [
                { s: "C", t: "Cursar disciplinas relacionadas ao doutorado." },
                { s: "C", t: "Levantamento inicial da literatura (Gêmeos Digitais, ITS)." },
                { s: "ED", t: "Participar de seminários e workshops técnicos." },
                { s: "C", t: "Estudo de ferramentas (Unity 3D, MongoDB)." }
            ]
        },
        "2": {
            title: "Ano 2 – Fase de Pesquisa e Coleta",
            activities: [
                { s: "ED", t: "Revisão exaustiva e definição do referencial teórico." },
                { s: "ED", t: "Identificação e coleta de dados de tráfego e infraestrutura." },
                { s: "ED", t: "Criação do plano conceitual do simulador SmartCitySystem." },
                { s: "C", t: "Redação do primeiro artigo científico preliminar." }
            ]
        },
        "3": {
            title: "Ano 3 – Desenvolvimento, Validação e Defesa",
            activities: [
                { s: "ED", t: "Implementação das funcionalidades de IA e sensores no Unity." },
                { s: "ED", t: "Simulação de cenários de mobilidade e segurança." },
                { s: "X", t: "Aplicação em estudos de caso reais e coleta de feedback." },
                { s: "X", t: "Redação final da tese e defesa do doutorado." }
            ]
        }
    };

    if (timelineItems.length > 0 && modalTimeline) {
        timelineItems.forEach(item => {
            item.addEventListener('click', () => {
                const year = item.getAttribute('data-year');
                const data = timelineData[year];

                let html = `
                    <div class="detail-year-header">
                        <h3 style="color: var(--neon-green); font-size: 24px;">${data.title}</h3>
                    </div>
                    <div class="activity-list">
                `;

                data.activities.forEach(act => {
                    html += `
                        <div class="activity-item">
                            <span class="activity-status status-${act.s.toLowerCase()}">${act.s}</span>
                            <span class="activity-text">${act.t}</span>
                        </div>
                    `;
                });

                html += `</div><p style="margin-top:20px; font-size:12px; color:#475569;">Legenda: C (Concluído), ED (Em Desenvolvimento), X (Planejado)</p>`;

                detailsContent.innerHTML = html;
                modalTimeline.classList.add('show');
            });
        });

        if(closeTimeline) {
            closeTimeline.addEventListener('click', () => modalTimeline.classList.remove('show'));
        }
    }
});