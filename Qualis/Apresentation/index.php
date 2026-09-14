<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gêmeos Digitais em Transporte Inteligente</title>

    <!-- Fontes -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Montserrat:wght@500;600;700;800&display=swap" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.4.0/css/all.min.css" rel="stylesheet">


    <link rel="stylesheet" href="css/style_core.css">
    <link rel="stylesheet" href="css/style_capa_obrigado.css">
    <link rel="stylesheet" href="css/style_sumario.css">
    <link rel="stylesheet" href="css/style_fundamentos.css">
    <link rel="stylesheet" href="css/style_proposta.css">
    <link rel="stylesheet" href="css/style_arquitetura.css">
    <link rel="stylesheet" href="css/style_sensores.css">
    <link rel="stylesheet" href="css/style_resultados.css">
    <link rel="stylesheet" href="css/style_cronograma.css">

</head>
<body>

<!-- Container Principal Escalável -->
<div id="presentation-area">

    <?php include 'slides/01_capa.php'; ?>
    <?php include 'slides/sumario.php'; ?>
    <?php include 'slides/01c_contexto.php'; ?>
    <?php include 'slides/02_problema.php'; ?>
    <?php include 'slides/03_motivacao.php'; ?>
    <?php include 'slides/04_objetivos.php'; ?>
    <?php include 'slides/05_pergunta_hipotese.php'; ?>
    <?php include 'slides/06_inovacao.php'; ?>
    <?php include 'slides/07_visao_simulador.php'; ?>
    <?php include 'slides/08_metodologia.php'; ?>
    <?php include 'slides/08b_fases.php'; ?>
    <?php include 'slides/09_arquitetura.php'; ?>
    <?php include 'slides/10_sensores.php'; ?>
    <?php include 'slides/11_integracao.php'; ?>
    <?php include 'slides/12_resultados.php'; ?>
    <?php include 'slides/12b_ambiente.php'; ?>
    <?php include 'slides/12c_dados.php'; ?>
    <?php include 'slides/12d_dashboard.php'; ?>
    <?php include 'slides/13_cenarios.php'; ?>
    <?php include 'slides/14_plano_trabalho.php'; ?>
    <?php include 'slides/14b_conclusao_parcial.php'; ?>
    <?php include 'slides/14c_futuro.php'; ?>
    <?php include 'slides/15_obrigado.php'; ?>

</div>

<!-- Controles da Apresentação -->
<div class="controls">
    <button class="control-btn" id="btn-prev" title="Anterior"><i class="fas fa-chevron-left"></i></button>

    <button class="control-btn" id="btn-summary" title="Voltar ao Roteiro" style="width: auto; padding: 0 20px; border-radius: 30px; font-size: 14px;">
        <i class="fas fa-map-marked-alt" style="margin-right: 8px;"></i> Roteiro
    </button>

    <button class="control-btn" id="btn-next" title="Próximo"><i class="fas fa-chevron-right"></i></button>
</div>

<div class="progress-container">
    <div class="progress-bar" id="progress-bar"></div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/animejs/3.2.2/anime.min.js"></script>
<script src="js/script_core.js"></script>
<script src="js/script_sumario.js"></script>
<script src="js/script_semaforo.js"></script>
<script src="js/script_modal_tabela.js"></script>
<script src="js/script_arquitetura.js"></script>
<script src="js/script_cronograma.js"></script>


</body>
</html>