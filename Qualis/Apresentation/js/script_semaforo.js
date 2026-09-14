document.addEventListener('DOMContentLoaded', () => {

    // =========================================================
    // LÓGICA DE TROCA DE ABAS (HOVER)
    // =========================================================
    const sensorCards = document.querySelectorAll('.sensor-card');
    const sensorVisuals = document.querySelectorAll('.sensor-visual');

    if (sensorCards.length > 0) {
        sensorCards.forEach(card => {
            card.addEventListener('mouseenter', () => {
                const target = card.getAttribute('data-target');

                // Atualiza o estado dos cartões da esquerda
                sensorCards.forEach(c => c.classList.remove('active-card'));
                card.classList.add('active-card');

                // Oculta todas as imagens e mostra apenas a correspondente
                sensorVisuals.forEach(v => v.classList.remove('active'));
                const targetVisual = document.getElementById('visual-' + target);
                if(targetVisual) {
                    targetVisual.classList.add('active');
                }
            });
        });
    }

    // =========================================================
    // LÓGICA DO VISUAL 1: SEMÁFORO
    // =========================================================
    const trafficLight = document.getElementById('interactive-light');
    if(trafficLight) {
        const lights = {
            red: document.getElementById('light-red'),
            yellow: document.getElementById('light-yellow'),
            green: document.getElementById('light-green')
        };
        let currentLightIndex = 0;
        const cycleOrder = ['red', 'yellow', 'green'];

        function switchLight(colorName) {
            Object.values(lights).forEach(el => el.classList.remove('active'));
            lights[colorName].classList.add('active');
        }

        setInterval(() => {
            // Só roda o semáforo se a aba dele estiver visível
            if(document.getElementById('visual-semaforo').classList.contains('active')) {
                currentLightIndex = (currentLightIndex + 1) % cycleOrder.length;
                switchLight(cycleOrder[currentLightIndex]);
            }
        }, 1500);
    }

    // =========================================================
    // LÓGICA DO VISUAL 2: CONTADOR DE VEÍCULOS (TRIGGER)
    // =========================================================
    let carCount = 0;
    const highway = document.getElementById('highway');
    const countDisplay = document.getElementById('car-count');
    const triggerBox = document.getElementById('trigger-box');

    if (highway && countDisplay && triggerBox && typeof anime !== 'undefined') {

        // Gera um carro a cada 1.8 segundos
        setInterval(() => {
            // Só gera carros se a aba do contador estiver visível
            if(document.getElementById('visual-contador').classList.contains('active')) {

                const car = document.createElement('div');
                car.classList.add('css-car');
                car.style.left = '-100px'; // Nasce fora da tela à esquerda

                // Cores aleatórias para os carros
                const colors = ['#f87171', '#60a5fa', '#34d399', '#facc15', '#f1f5f9', '#a78bfa'];
                car.style.background = colors[Math.floor(Math.random() * colors.length)];

                // Varia a faixa da pista (cima ou baixo)
                car.style.top = Math.random() > 0.5 ? '30%' : '60%';

                highway.appendChild(car);

                // Anima o carro atravessando a rua usando Anime.js
                anime({
                    targets: car,
                    left: '500px', // Vai até fora da tela à direita
                    duration: 2500,
                    easing: 'linear',
                    update: function(anim) {
                        const currentLeft = parseInt(car.style.left);

                        // Verifica se o carro entrou no Trigger (aproximadamente no meio da tela)
                        if (currentLeft > 180 && currentLeft < 220 && !car.counted) {
                            car.counted = true; // Marca para não contar duas vezes

                            // Formata o número com zero à esquerda (ex: 01, 02)
                            carCount++;
                            countDisplay.innerText = carCount < 10 ? '0' + carCount : carCount;

                            // Faz a caixa do Trigger piscar em Neon
                            triggerBox.classList.add('flash');
                            setTimeout(() => triggerBox.classList.remove('flash'), 150);
                        }
                    },
                    complete: function() {
                        car.remove(); // Destrói o elemento HTML para não pesar o navegador
                    }
                });
            }
        }, 1800);
    }
});