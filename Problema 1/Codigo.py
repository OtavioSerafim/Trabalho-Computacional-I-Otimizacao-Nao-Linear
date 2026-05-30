import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize


class ProblemaMassaMola:
    """Problema de otimização do quociente de Rayleigh para sistema massa-mola."""

    def __init__(self):
        self.M = np.eye(3)
        self.K = np.array([[2, -1, 0],
                           [-1, 2, -1],
                           [0, -1, 1]], dtype=float)

    def funcaoobjetivo(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        numerador = x @ self.K @ x
        denominador = x @ self.M @ x
        return numerador / denominador


class Solution:
    def __init__(self, x=None, fx=None, iter=None, aval=None, xhist=None,
                 fxhist=None):
        self.x = x.reshape(x.size)
        self.fx = fx
        self.iter = iter
        self.aval = aval
        self.xhist = xhist
        self.fxhist = fxhist

    def resultados(self, func, xlim, ylim, x3_fix, levels=None):
        x1, x2 = np.meshgrid(np.linspace(xlim[0], xlim[1]),
                             np.linspace(ylim[0], ylim[1]))

        f = np.zeros(x1.shape)
        for i in range(x1.shape[0]):
            for j in range(x1.shape[1]):
                f[i, j] = func(np.array([x1[i, j], x2[i, j], x3_fix]))

        _, axis = plt.subplots(ncols=2, figsize=[2 * 6.4, 4.8])
        if levels is not None:
            axis[0].contour(x1, x2, f, levels=levels)
        else:
            axis[0].contour(x1, x2, f, levels=30)
        axis[0].plot(self.xhist[:, 0], self.xhist[:, 1], '--*r')
        axis[0].plot(self.x[0], self.x[1], '*g', label='ótimo')
        axis[0].set_xlabel(r'$x_1$')
        axis[0].set_ylabel(r'$x_2$')
        axis[0].set_title('Problema')
        axis[0].legend()
        axis[0].grid()

        axis[1].plot(self.fxhist, '--*')
        axis[1].set_xlabel('Iterações')
        axis[1].set_ylabel(r'$R(\mathbf{x})$')
        axis[1].set_title('Convergência')
        axis[1].grid()

        plt.tight_layout()
        plt.show()

    def __str__(self):
        mensagem = ''
        mensagem += 'Solução ótima: ' + str(self.x) + '\n'
        mensagem += 'Número de iterações: %d\n' % self.iter
        mensagem += 'Número de avaliações: %d' % self.aval
        return mensagem


class QuasiNewton:
    """Método Quasi-Newton (BFGS) via scipy.optimize.minimize."""

    def __init__(self, maximo_iteracoes=1000, maximo_avaliacoes=15000):
        self.maximo_iteracoes = maximo_iteracoes
        self.maximo_avaliacoes = maximo_avaliacoes

    def resolva(self, func, solucao_inicial):
        xhist = [np.asarray(solucao_inicial, dtype=float).reshape(-1)]
        fxhist = [func(xhist[0])]

        def callback(xk):
            xk = np.asarray(xk, dtype=float).reshape(-1)
            xhist.append(xk.copy())
            fxhist.append(func(xk))

        resultado = minimize(func, solucao_inicial, method='BFGS',
                             callback=callback,
                             options={'maxiter': self.maximo_iteracoes})

        solucao = Solution(
            x=resultado.x,
            fx=resultado.fun,
            iter=resultado.nit,
            aval=resultado.nfev,
            xhist=np.array(xhist),
            fxhist=np.array(fxhist),
        )
        return solucao

if __name__ == '__main__':
    problema = ProblemaMassaMola()

    maximo_iteracoes = 1000
    maximo_avaliacoes = 15000
    solucao_inicial = [1, 1, 1]

    metodo = QuasiNewton(maximo_iteracoes=maximo_iteracoes,
                         maximo_avaliacoes=maximo_avaliacoes)
    solucao = metodo.resolva(problema.funcaoobjetivo, solucao_inicial)

    solucao_final = solucao.x
    funcaoobjetivo_final = solucao.fx
    numero_iteracoes = solucao.iter
    numero_avaliacoes = solucao.aval

    print(f'Vetor x ótimo: {solucao_final}')
    print(f'Função-objetivo final R(x*): {funcaoobjetivo_final}')
    print(f'Número de iterações: {numero_iteracoes}')
    print(f'Número de avaliações: {numero_avaliacoes}')

    x3_fix = solucao_inicial[2]
    margem = 0.5
    xlim = [solucao.xhist[:, 0].min() - margem,
            solucao.xhist[:, 0].max() + margem]
    ylim = [solucao.xhist[:, 1].min() - margem,
            solucao.xhist[:, 1].max() + margem]

    solucao.resultados(problema.funcaoobjetivo, xlim, ylim, x3_fix)
