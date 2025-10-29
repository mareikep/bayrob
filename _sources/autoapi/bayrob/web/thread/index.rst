bayrob.web.thread
=================

.. py:module:: bayrob.web.thread


Classes
-------

.. autoapisummary::

   bayrob.web.thread.BayRoBSessionThread


Module Contents
---------------

.. py:class:: BayRoBSessionThread(webapp)

   Bases: :py:obj:`pyrap.threads.DetachedSessionThread`


   .. py:attribute:: pushsession


   .. py:attribute:: webapp


   .. py:attribute:: callback
      :value: None



   .. py:attribute:: runfunction
      :value: 'queryjpt'



   .. py:property:: query
      :type: bayrob.core.base.Query



   .. py:property:: models
      :type: dict



   .. py:property:: datasets
      :type: dict



   .. py:method:: adddatapath(path) -> None


   .. py:property:: result
      :type: bayrob.core.base.BayRoB.Result



   .. py:method:: query_jpts() -> None


   .. py:method:: astar() -> None


   .. py:method:: run() -> None


