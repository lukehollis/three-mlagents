import { createStore, combineReducers, applyMiddleware } from 'redux';
import thunk from 'redux-thunk';
// import { keplerGlReducer } from '@kepler.gl/reducers';

// Temporary simple reducer to replace kepler.gl
const simpleReducer = (state = {}, action) => {
  return state;
};

const reducers = combineReducers({
  simple: simpleReducer,
});

const store = createStore(reducers, {}, applyMiddleware(thunk));

export default store;