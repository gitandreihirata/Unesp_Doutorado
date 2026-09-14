document.addEventListener('DOMContentLoaded', () => {
    const roadPoints = document.querySelectorAll('.road-point');
    const cyberCar = document.getElementById('cyber-car');
    const carBody = document.getElementById('car-body');

    if (roadPoints.length > 0 && cyberCar && typeof anime !== 'undefined') {
        let currentIndex = -1;

        roadPoints.forEach(point => {
            // Evento 1: Hover (Animar o carrinho)
            point.addEventListener('mouseenter', function() {
                const targetIndex = parseInt(this.getAttribute('data-index'));

                if(currentIndex === -1) {
                    currentIndex = targetIndex;
                    cyberCar.style.left = this.style.left;
                    cyberCar.style.top = this.style.top;
                    cyberCar.style.opacity = '1';
                    return;
                }

                if (currentIndex === targetIndex) return;

                const isGoingForward = targetIndex > currentIndex;
                carBody.style.transform = isGoingForward ? 'scaleX(1)' : 'scaleX(-1)';

                let pathKeyframes = [];
                const step = isGoingForward ? 1 : -1;

                for (let i = currentIndex + step; i !== targetIndex + step; i += step) {
                    const nextPoint = document.querySelector(`.road-point[data-index="${i}"]`);
                    pathKeyframes.push({ left: nextPoint.style.left, top: nextPoint.style.top });
                }

                const travelDuration = Math.abs(targetIndex - currentIndex) * 300;

                anime.remove(cyberCar);
                anime({
                    targets: cyberCar,
                    keyframes: pathKeyframes,
                    duration: travelDuration,
                    easing: 'linear',
                    begin: function() {
                        cyberCar.classList.add('moving');
                        cyberCar.style.opacity = '1';
                    },
                    complete: function() {
                        cyberCar.classList.remove('moving');
                        currentIndex = targetIndex;
                    }
                });
            });
        });

        // Esconder carrinho se o mouse sair do container inteiro
        const roadmapContainer = document.querySelector('.roadmap-container');
        if(roadmapContainer) {
            roadmapContainer.addEventListener('mouseleave', () => {
                cyberCar.classList.remove('moving');
                anime({
                    targets: cyberCar,
                    opacity: 0,
                    duration: 500,
                    easing: 'linear',
                    complete: function() { currentIndex = -1; }
                });
            });
        }
    }
});