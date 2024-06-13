import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
from sklearn.decomposition import PCA
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go


class Portfolio:


    def __init__(self,data:pd.DataFrame,index:pd.DataFrame,year_conv:int = 252):
        
        """
        Represents a portfolio of stocks, for which is to be used to construct an eigenportfolio.

        Attributes
        ----------
            data: pd.DataFrame
                A dataset of stocks, which represent the investible universe for the portfolio.
            
            index: pd.DataFrame
                A dataset which represents an index for the given data, ex. SPY
            
            year_conv: int
                The year convention to assume when annualizing statistics.

            stats: defaultdict(dict)
                A dictionary of eigenportfolio specific statistics
            
            weights: np.array
                An array of weights corresponding to normalized eigenvectors
            
            pca: sklearn.PCA
                The PCA object fit to the data.
        """

        self.data = data.copy()
        self.index = index.copy()
        self.year_conv = year_conv
        self.stats = defaultdict(dict)
        self.weights = None
        self.pca = None

    def fitPortfolios(self,n_components:int = 1) -> None:

        """
        Compute PCA on the given data and store weights.

        Inputs
        ------
            n_components: int
                The number of principal components to fit with PCA.
        """

        if n_components > self.data.shape[1]:
            raise ValueError("Can only fit a maxmimum of n components for a m*n matrix.")

        # compute log returns
        self.returns = self._getReturns(self.data)

        # PCA
        self.pca = PCA(n_components)
        self.pca = self.pca.fit(self.returns)

        # Retrieve and Normalize Weights
        self.weights = self._normalize(self.pca.components_)

        return 

    def getPortfolioStats(self,portfolio_n = 0):

        """
        
        Returns a dictionary of summary statistics for the selected portfolio

        Inputs
        ------
            portfolio_n: int
                An integer corresponding to which principal component to use as weights

        Returns
        -------
            dict
                A dictionary of portfolio statistics
        """

        if len(self.stats) != 0:
            return self.stats[portfolio_n]
        
        for n in range(len(self.weights)):

            # Compute Portfolio Weighted Returns
            returns = np.exp((self.returns*self.weights[n]).sum(axis = 1))
            # Portfolio Stats
            ret,vol,sharpe = self._computesStats(returns)

            self.stats[n] = {'ret_ts':returns,'annual_ret':ret,
                             'annual_vol':vol,'annual_sharpe':sharpe}

        return self.stats[portfolio_n]
    
    def plotPortfolioPerformance(self,portfolio_n):
        """
        Plots the performance of the portfolio over the given period.

        Inputs
        ------
            portfolio_n: int
                An integer corresponding to the principal component vector to be used as weights

        Returns
        -------
            plotly.Figure
                A plotly graph of portfolio performance.
        """

        _portfolio_ts = self.stats[portfolio_n]['ret_ts'].cumprod()
        _index_ts = np.exp(self._getReturns(self.index)).cumprod()

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                name = f'Eigenportfolio {portfolio_n}',
                x = _portfolio_ts.index,
                y = _portfolio_ts.values
            )
        )

        fig.add_trace(
            go.Scatter(
                name = 'Index',
                x = _index_ts.index,
                y = _index_ts.iloc[:,0],
            )
        )
        fig.update_layout(title = f'Eigenportfolio {portfolio_n} Performance vs Index')
        fig.update_yaxes(title = 'Cumulative Return')
        return fig



    def plotPortfolioWeights(self,portfolio_n):
        """
        Plots the portfolio weightings for a given eigenportfolio

        Inputs
        ------
            portfolio_n: int
                An integer corresponding to the principal component vector to be used as weights

        Returns
        -------
            plotly.Figure
                A plotly graph of portfolio weights
        """

        df = pd.DataFrame()
        df['Ticker'] = self.data.columns
        df['Weight'] = self.weights[portfolio_n]
        df["Color"] = np.where(df["Weight"]<0, '#FFCCCB', 'lightgreen')

        fig = px.bar(
            df,
            x = 'Ticker',
            y = 'Weight',

        )
        fig.update_layout(title = f'Eigenportfolio {portfolio_n} Weight Attribution')
        fig.update_traces(marker_color=df["Color"])

        return fig


    def _getReturns(self,data) -> pd.DataFrame:
        """
        Computes log returns for a given price time-series.

        Inputs
        ------
            data: pd.DataFrame
                A price time series
        
        Returns
        -------
            pd.DataFrame:
                A price time series of log returns
        
        """
        return (np.log(data / data.shift(1))).dropna()
    

    def _normalize(self,weights) -> np.array:

        """
        Normalizes a np.array vector such that it sums to 1.

        Inputs 
        ------
            weights: np.array
                An np.array vector of weights.

        Returns
        -------
            np.array
                An np.array vector of weights that sum to 1.
        """

        return weights/weights.sum(axis=1, keepdims=True)
        
    def _computesStats(self,returns) -> tuple[float]:

        """
        Computes portfolio relevant stats

        Inputs
        ------
            returns: pd.DataFrame
                A dataset of stock log-returns

        Returns
        -------
            tuple
                A tuple with annualized returns,annualized volatility, and sharpe ratio
        """

        num_years = returns.shape[0] / self.year_conv
        ann_return = returns.prod()**(1/num_years) - 1
        ann_volatility = np.std((np.exp(returns) - 1)) * np.sqrt(self.year_conv)
        ann_sharpe = ann_return / ann_volatility

        return ann_return,ann_volatility,ann_sharpe