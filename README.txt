# notebooks_MachineLearning
Vou estar fazendo o upload de alguns testes com ML em python
"""
Sistema de Alocação de Membros em Squads
========================================
Resolve dois problemas:
1. Redistribuição de membros entre squads de uma linha
2. Alocação de pessoas desalocadas em squads com déficit

Usa Google OR-Tools (CP-SAT) para otimização com restrições.

Instalação: pip install ortools pandas
"""

from ortools.sat.python import cp_model
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# 1. ESTRUTURAS DE DADOS
# =============================================================================

@dataclass
class Pessoa:
    id: str
    nome: str
    papel: str  # dev, qa, data_engineer, data_scientist, tech_lead, etc.
    senioridade: str  # junior, pleno, senior, especialista
    skills: dict  # skill -> nível (1-5, do formulário de autopercepção)
    squad_atual: str = None  # None se desalocado
    linha_atual: str = None


@dataclass
class Squad:
    id: str
    nome: str
    linha: str
    capacidade_max: int
    capacidade_min: int
    necessidades: dict  # papel -> quantidade mínima necessária
    skills_desejadas: dict  # skill -> nível mínimo desejado


@dataclass
class Restricao:
    """Restrições parametrizáveis"""
    tipo: str  # 'bloquear', 'forcar', 'preferencia'
    pessoa_id: str = None
    squad_id: str = None
    linha_id: str = None
    peso: int = 1  # para preferências


# =============================================================================
# 2. DADOS DE EXEMPLO (simula seu cenário)
# =============================================================================

def criar_dados_exemplo():
    """Cria dados fictícios para demonstração"""
    
    # ----- PESSOAS -----
    pessoas = [
        # Pessoas alocadas na Linha Alpha
        Pessoa("p1", "Ana Silva", "dev", "senior", 
               {"python": 5, "java": 3, "react": 4, "aws": 4}, 
               "squad_a1", "alpha"),
        Pessoa("p2", "Bruno Costa", "dev", "pleno", 
               {"python": 4, "java": 4, "react": 2, "aws": 2}, 
               "squad_a1", "alpha"),
        Pessoa("p3", "Carla Dias", "qa", "senior", 
               {"automacao": 5, "python": 3, "cypress": 5}, 
               "squad_a1", "alpha"),
        Pessoa("p4", "Diego Alves", "data_engineer", "pleno", 
               {"python": 4, "spark": 4, "sql": 5, "aws": 3}, 
               "squad_a2", "alpha"),
        Pessoa("p5", "Elena Rocha", "dev", "junior", 
               {"python": 3, "java": 2, "react": 3}, 
               "squad_a2", "alpha"),
        Pessoa("p6", "Felipe Lima", "qa", "pleno", 
               {"automacao": 4, "cypress": 4, "selenium": 3}, 
               "squad_a2", "alpha"),
        Pessoa("p7", "Gabi Nunes", "data_scientist", "senior", 
               {"python": 5, "ml": 5, "sql": 4, "spark": 3}, 
               "squad_a3", "alpha"),
        Pessoa("p8", "Hugo Martins", "dev", "senior", 
               {"python": 4, "java": 5, "react": 3, "aws": 5}, 
               "squad_a3", "alpha"),
        
        # Pessoas DESALOCADAS (sem squad)
        Pessoa("p9", "Iris Campos", "dev", "pleno", 
               {"python": 4, "react": 5, "typescript": 4}, 
               None, None),
        Pessoa("p10", "João Pedro", "qa", "junior", 
               {"automacao": 3, "cypress": 3, "python": 2}, 
               None, None),
        Pessoa("p11", "Karen Souza", "data_engineer", "senior", 
               {"python": 5, "spark": 5, "sql": 5, "aws": 4, "kafka": 4}, 
               None, None),
        Pessoa("p12", "Lucas Ferreira", "dev", "pleno", 
               {"java": 5, "spring": 4, "aws": 3}, 
               None, None),
        Pessoa("p13", "Mariana Oliveira", "data_scientist", "pleno", 
               {"python": 4, "ml": 4, "sql": 3, "estatistica": 4}, 
               None, None),
    ]
    
    # ----- SQUADS -----
    squads = [
        # Linha Alpha
        Squad("squad_a1", "Squad Pagamentos", "alpha", 
              capacidade_max=5, capacidade_min=3,
              necessidades={"dev": 2, "qa": 1},
              skills_desejadas={"python": 3, "aws": 3}),
        Squad("squad_a2", "Squad Checkout", "alpha", 
              capacidade_max=5, capacidade_min=3,
              necessidades={"dev": 2, "qa": 1, "data_engineer": 1},
              skills_desejadas={"python": 3, "react": 3}),
        Squad("squad_a3", "Squad Analytics", "alpha", 
              capacidade_max=4, capacidade_min=2,
              necessidades={"dev": 1, "data_scientist": 1},
              skills_desejadas={"python": 4, "ml": 3, "spark": 3}),
        
        # Linha Beta (para alocação de desalocados)
        Squad("squad_b1", "Squad Mobile", "beta", 
              capacidade_max=5, capacidade_min=3,
              necessidades={"dev": 3, "qa": 1},
              skills_desejadas={"react": 4, "typescript": 3}),
        Squad("squad_b2", "Squad Data Platform", "beta", 
              capacidade_max=4, capacidade_min=2,
              necessidades={"data_engineer": 2, "data_scientist": 1},
              skills_desejadas={"spark": 4, "python": 4, "kafka": 3}),
    ]
    
    # ----- RESTRIÇÕES PARAMETRIZÁVEIS -----
    restricoes = [
        # Bruno não pode ir para Squad Analytics (exemplo: conflito pessoal)
        Restricao("bloquear", pessoa_id="p2", squad_id="squad_a3"),
        # Karen deve ir para Squad Data Platform (exemplo: projeto estratégico)
        Restricao("forcar", pessoa_id="p11", squad_id="squad_b2"),
    ]
    
    return pessoas, squads, restricoes


# =============================================================================
# 3. FUNÇÕES DE SCORING
# =============================================================================

def calcular_fit_skill(pessoa: Pessoa, squad: Squad) -> int:
    """
    Calcula score de fit baseado nas skills da pessoa vs necessidades da squad.
    Retorna valor de 0-100.
    """
    if not squad.skills_desejadas:
        return 50  # neutro se squad não tem preferências
    
    score = 0
    max_score = 0
    
    for skill, nivel_minimo in squad.skills_desejadas.items():
        max_score += 5  # máximo possível por skill
        nivel_pessoa = pessoa.skills.get(skill, 0)
        
        if nivel_pessoa >= nivel_minimo:
            # Bonus por atender ou exceder
            score += min(nivel_pessoa, 5)
        else:
            # Penalidade por não atender
            score += nivel_pessoa * 0.5
    
    if max_score == 0:
        return 50
    
    return int((score / max_score) * 100)


def calcular_fit_papel(pessoa: Pessoa, squad: Squad) -> int:
    """
    Calcula score baseado se o papel da pessoa é necessário na squad.
    Retorna valor de 0-100.
    """
    if pessoa.papel in squad.necessidades:
        return 100
    return 30  # ainda pode contribuir, mas não é ideal


def calcular_fit_total(pessoa: Pessoa, squad: Squad) -> int:
    """
    Combina diferentes fatores de fit.
    Pesos podem ser ajustados conforme necessidade.
    """
    peso_skill = 0.6
    peso_papel = 0.4
    
    fit_skill = calcular_fit_skill(pessoa, squad)
    fit_papel = calcular_fit_papel(pessoa, squad)
    
    return int(fit_skill * peso_skill + fit_papel * peso_papel)


# =============================================================================
# 4. SOLVER - REDISTRIBUIÇÃO INTRA-LINHA
# =============================================================================

def redistribuir_linha(pessoas: list, squads: list, 
                       linha: str, restricoes: list = None) -> dict:
    """
    Redistribui membros entre squads de uma mesma linha para balancear skills.
    
    Args:
        pessoas: Lista de todas as pessoas
        squads: Lista de todas as squads
        linha: ID da linha a ser redistribuída
        restricoes: Lista de restrições parametrizáveis
    
    Returns:
        Dicionário com alocações {pessoa_id: squad_id}
    """
    
    # Filtrar pessoas e squads da linha
    pessoas_linha = [p for p in pessoas if p.linha_atual == linha]
    squads_linha = [s for s in squads if s.linha == linha]
    
    if not pessoas_linha or not squads_linha:
        print(f"Linha '{linha}' sem pessoas ou squads para redistribuir.")
        return {}
    
    print(f"\n{'='*60}")
    print(f"REDISTRIBUIÇÃO - LINHA: {linha.upper()}")
    print(f"{'='*60}")
    print(f"Pessoas: {len(pessoas_linha)} | Squads: {len(squads_linha)}")
    
    # Criar modelo
    model = cp_model.CpModel()
    
    # ----- VARIÁVEIS -----
    # x[pessoa_id, squad_id] = 1 se pessoa alocada na squad
    x = {}
    for p in pessoas_linha:
        for s in squads_linha:
            x[p.id, s.id] = model.NewBoolVar(f'x_{p.id}_{s.id}')
    
    # ----- RESTRIÇÕES BÁSICAS -----
    
    # Cada pessoa em exatamente 1 squad
    for p in pessoas_linha:
        model.Add(sum(x[p.id, s.id] for s in squads_linha) == 1)
    
    # Respeitar capacidade das squads
    for s in squads_linha:
        membros = sum(x[p.id, s.id] for p in pessoas_linha)
        model.Add(membros <= s.capacidade_max)
        model.Add(membros >= s.capacidade_min)
    
    # ----- RESTRIÇÕES PARAMETRIZÁVEIS -----
    if restricoes:
        for r in restricoes:
            if r.tipo == "bloquear" and r.pessoa_id and r.squad_id:
                if (r.pessoa_id, r.squad_id) in x:
                    model.Add(x[r.pessoa_id, r.squad_id] == 0)
                    print(f"  [Restrição] Bloqueado: {r.pessoa_id} → {r.squad_id}")
            
            elif r.tipo == "forcar" and r.pessoa_id and r.squad_id:
                if (r.pessoa_id, r.squad_id) in x:
                    model.Add(x[r.pessoa_id, r.squad_id] == 1)
                    print(f"  [Restrição] Forçado: {r.pessoa_id} → {r.squad_id}")
    
    # ----- OBJETIVO -----
    # Maximizar fit total + bonus por manter alocação atual (reduz churn)
    
    objetivo = []
    BONUS_PERMANENCIA = 10  # bonus por não mudar de squad
    
    for p in pessoas_linha:
        for s in squads_linha:
            fit = calcular_fit_total(p, s)
            
            # Bonus se pessoa já está nessa squad
            if p.squad_atual == s.id:
                fit += BONUS_PERMANENCIA
            
            objetivo.append(fit * x[p.id, s.id])
    
    model.Maximize(sum(objetivo))
    
    # ----- RESOLVER -----
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30  # timeout
    status = solver.Solve(model)
    
    # ----- EXTRAIR RESULTADOS -----
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        resultado = {}
        mudancas = 0
        
        print(f"\nStatus: {'ÓTIMO' if status == cp_model.OPTIMAL else 'VIÁVEL'}")
        print(f"\n{'Pessoa':<20} {'Papel':<15} {'De':<20} {'Para':<20} {'Fit':>5}")
        print("-" * 85)
        
        for p in pessoas_linha:
            for s in squads_linha:
                if solver.Value(x[p.id, s.id]) == 1:
                    resultado[p.id] = s.id
                    mudou = "→ MUDOU" if p.squad_atual != s.id else ""
                    if p.squad_atual != s.id:
                        mudancas += 1
                    fit = calcular_fit_total(p, s)
                    print(f"{p.nome:<20} {p.papel:<15} {p.squad_atual or 'N/A':<20} {s.id:<20} {fit:>5} {mudou}")
        
        print(f"\nTotal de mudanças: {mudancas}")
        print(f"Score total: {int(solver.ObjectiveValue())}")
        
        return resultado
    else:
        print("❌ Não foi possível encontrar solução viável!")
        print("   Verifique se as restrições não são conflitantes.")
        return {}


# =============================================================================
# 5. SOLVER - ALOCAÇÃO DE DESALOCADOS
# =============================================================================

def alocar_desalocados(pessoas: list, squads: list,
                       restricoes: list = None) -> dict:
    """
    Aloca pessoas desalocadas nas squads com déficit.
    
    Args:
        pessoas: Lista de todas as pessoas
        squads: Lista de todas as squads
        restricoes: Lista de restrições parametrizáveis
    
    Returns:
        Dicionário com alocações {pessoa_id: squad_id}
    """
    
    # Filtrar pessoas desalocadas
    desalocados = [p for p in pessoas if p.squad_atual is None]
    
    if not desalocados:
        print("Não há pessoas desalocadas.")
        return {}
    
    # Calcular vagas disponíveis por squad
    alocados_por_squad = {}
    for p in pessoas:
        if p.squad_atual:
            alocados_por_squad[p.squad_atual] = alocados_por_squad.get(p.squad_atual, 0) + 1
    
    squads_com_vaga = []
    for s in squads:
        atual = alocados_por_squad.get(s.id, 0)
        vagas = s.capacidade_max - atual
        if vagas > 0:
            squads_com_vaga.append((s, vagas))
    
    if not squads_com_vaga:
        print("Não há vagas disponíveis em nenhuma squad.")
        return {}
    
    print(f"\n{'='*60}")
    print("ALOCAÇÃO DE DESALOCADOS")
    print(f"{'='*60}")
    print(f"Pessoas desalocadas: {len(desalocados)}")
    print(f"Squads com vagas: {len(squads_com_vaga)}")
    print("\nVagas disponíveis:")
    for s, vagas in squads_com_vaga:
        print(f"  - {s.nome} ({s.linha}): {vagas} vaga(s)")
    
    # Criar modelo
    model = cp_model.CpModel()
    
    # ----- VARIÁVEIS -----
    x = {}
    for p in desalocados:
        for s, _ in squads_com_vaga:
            x[p.id, s.id] = model.NewBoolVar(f'x_{p.id}_{s.id}')
    
    # ----- RESTRIÇÕES BÁSICAS -----
    
    # Cada pessoa em no máximo 1 squad (pode ficar sem se não houver fit)
    for p in desalocados:
        model.Add(sum(x[p.id, s.id] for s, _ in squads_com_vaga) <= 1)
    
    # Respeitar vagas disponíveis
    for s, vagas in squads_com_vaga:
        model.Add(sum(x[p.id, s.id] for p in desalocados) <= vagas)
    
    # ----- RESTRIÇÕES PARAMETRIZÁVEIS -----
    if restricoes:
        for r in restricoes:
            if r.tipo == "bloquear" and r.pessoa_id and r.squad_id:
                if (r.pessoa_id, r.squad_id) in x:
                    model.Add(x[r.pessoa_id, r.squad_id] == 0)
                    print(f"  [Restrição] Bloqueado: {r.pessoa_id} → {r.squad_id}")
            
            elif r.tipo == "forcar" and r.pessoa_id and r.squad_id:
                if (r.pessoa_id, r.squad_id) in x:
                    model.Add(x[r.pessoa_id, r.squad_id] == 1)
                    print(f"  [Restrição] Forçado: {r.pessoa_id} → {r.squad_id}")
            
            elif r.tipo == "bloquear" and r.pessoa_id and r.linha_id:
                # Bloquear pessoa de toda uma linha
                for s, _ in squads_com_vaga:
                    if s.linha == r.linha_id and (r.pessoa_id, s.id) in x:
                        model.Add(x[r.pessoa_id, s.id] == 0)
                print(f"  [Restrição] Bloqueado: {r.pessoa_id} → linha {r.linha_id}")
    
    # ----- OBJETIVO -----
    # Maximizar fit total + priorizar alocar o máximo de pessoas
    
    BONUS_ALOCACAO = 50  # bonus por alocar alguém (prioriza não deixar desalocado)
    
    objetivo = []
    for p in desalocados:
        for s, _ in squads_com_vaga:
            fit = calcular_fit_total(p, s) + BONUS_ALOCACAO
            objetivo.append(fit * x[p.id, s.id])
    
    model.Maximize(sum(objetivo))
    
    # ----- RESOLVER -----
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    status = solver.Solve(model)
    
    # ----- EXTRAIR RESULTADOS -----
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        resultado = {}
        alocados_count = 0
        
        print(f"\nStatus: {'ÓTIMO' if status == cp_model.OPTIMAL else 'VIÁVEL'}")
        print(f"\n{'Pessoa':<20} {'Papel':<15} {'Alocado em':<25} {'Linha':<10} {'Fit':>5}")
        print("-" * 80)
        
        for p in desalocados:
            alocado = False
            for s, _ in squads_com_vaga:
                if solver.Value(x[p.id, s.id]) == 1:
                    resultado[p.id] = s.id
                    fit = calcular_fit_total(p, s)
                    print(f"{p.nome:<20} {p.papel:<15} {s.nome:<25} {s.linha:<10} {fit:>5}")
                    alocados_count += 1
                    alocado = True
                    break
            
            if not alocado:
                print(f"{p.nome:<20} {p.papel:<15} {'❌ Não alocado':<25}")
        
        print(f"\nAlocados: {alocados_count}/{len(desalocados)}")
        print(f"Score total: {int(solver.ObjectiveValue())}")
        
        return resultado
    else:
        print("❌ Não foi possível encontrar solução viável!")
        return {}


# =============================================================================
# 6. RELATÓRIO DE DIAGNÓSTICO
# =============================================================================

def gerar_diagnostico(pessoas: list, squads: list):
    """Gera relatório de diagnóstico da situação atual"""
    
    print(f"\n{'='*60}")
    print("DIAGNÓSTICO DA SITUAÇÃO ATUAL")
    print(f"{'='*60}")
    
    # Contagem por squad
    print("\n📊 OCUPAÇÃO DAS SQUADS:")
    print("-" * 50)
    
    for s in squads:
        membros = [p for p in pessoas if p.squad_atual == s.id]
        ocupacao = len(membros)
        status = "✅" if s.capacidade_min <= ocupacao <= s.capacidade_max else "⚠️"
        
        print(f"\n{status} {s.nome} ({s.linha})")
        print(f"   Ocupação: {ocupacao}/{s.capacidade_max} (mín: {s.capacidade_min})")
        
        # Verificar necessidades de papéis
        papeis_presentes = {}
        for m in membros:
            papeis_presentes[m.papel] = papeis_presentes.get(m.papel, 0) + 1
        
        for papel, qtd_necessaria in s.necessidades.items():
            qtd_presente = papeis_presentes.get(papel, 0)
            check = "✓" if qtd_presente >= qtd_necessaria else "✗"
            print(f"   [{check}] {papel}: {qtd_presente}/{qtd_necessaria}")
    
    # Pessoas desalocadas
    desalocados = [p for p in pessoas if p.squad_atual is None]
    if desalocados:
        print(f"\n⚠️  PESSOAS DESALOCADAS: {len(desalocados)}")
        print("-" * 50)
        for p in desalocados:
            skills_str = ", ".join([f"{k}:{v}" for k, v in list(p.skills.items())[:3]])
            print(f"   • {p.nome} ({p.papel}, {p.senioridade}) - {skills_str}...")


# =============================================================================
# 7. MAIN - EXECUÇÃO
# =============================================================================

def main():
    # Carregar dados
    pessoas, squads, restricoes = criar_dados_exemplo()
    
    # Diagnóstico inicial
    gerar_diagnostico(pessoas, squads)
    
    # Problema 1: Redistribuição da linha Alpha
    resultado_redistribuicao = redistribuir_linha(
        pessoas, squads, 
        linha="alpha", 
        restricoes=restricoes
    )
    
    # Problema 2: Alocar desalocados
    resultado_alocacao = alocar_desalocados(
        pessoas, squads,
        restricoes=restricoes
    )
    
    # Resumo final
    print(f"\n{'='*60}")
    print("RESUMO FINAL")
    print(f"{'='*60}")
    print(f"Redistribuições na linha Alpha: {len(resultado_redistribuicao)} pessoas")
    print(f"Novas alocações: {len(resultado_alocacao)} pessoas")


if __name__ == "__main__":
    main()
